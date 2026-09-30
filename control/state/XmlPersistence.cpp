#include "XmlPersistence.h"

#include <algorithm>

namespace spatcore::control::state
{

//==============================================================================
// Construction
//==============================================================================

XmlPersistence::XmlPersistence (Options optionsIn)
    : options (std::move (optionsIn))
{
}

//==============================================================================
// XML File I/O
//==============================================================================

juce::String XmlPersistence::makeXmlHeader (const juce::String& fileType) const
{
    juce::String header;
    header << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    header << "<!-- " << options.headerTitle << " -->\n";
    header << "<!-- Type: " << fileType << " -->\n";
    header << "<!-- Created: " << juce::Time::getCurrentTime().toString (true, true) << " -->\n";
    header << "\n";
    return header;
}

XmlPersistence::WriteResult XmlPersistence::writeTreeToFile (const juce::ValueTree& tree,
                                                             const juce::File& file) const
{
    auto xml = tree.createXml();
    if (xml == nullptr)
        return WriteResult::xmlConversionFailed;

    // Create human-readable XML with the custom header (without JUCE's default declaration)
    juce::String header = makeXmlHeader (file.getFileNameWithoutExtension());
    auto format = juce::XmlElement::TextFormat().withoutHeader();
    juce::String xmlString = header + xml->toString (format);

    // The bytes File::replaceWithText used to write (UTF-8, no byte-order
    // mark, CRLF line ends), so saved files do not change.
    juce::MemoryOutputStream bytes;
    bytes.writeText (xmlString, false, false, "\r\n");

    if (!replaceFileContents (file, bytes.getData(), bytes.getDataSize()))
        return WriteResult::fileWriteFailed;

    return WriteResult::ok;
}

bool XmlPersistence::replaceFileContents (const juce::File& file, const void* data, size_t numBytes)
{
    juce::TemporaryFile temp (file, juce::TemporaryFile::useHiddenFile);

    {
        juce::FileOutputStream out (temp.getFile());
        if (out.failedToOpen())
            return false;

        if (numBytes > 0 && !out.write (data, numBytes))
            return false;

        // The stream buffers what it is given; a full disk or a pulled drive
        // shows up when the buffer goes out and the file is synced, which
        // flush() does here and the destructor would do without telling us.
        out.flush();
        if (out.getStatus().failed())
            return false;
    }

    if (temp.getFile().getSize() != static_cast<juce::int64> (numBytes))
        return false;

    // Only now does the complete file take the old one's place. On failure
    // TemporaryFile's destructor deletes what was written.
    return temp.overwriteTargetFileWithTemporary();
}

XmlPersistence::ReadResult XmlPersistence::readTreeFromFile (const juce::File& file) const
{
    if (!file.existsAsFile())
        return { {}, ReadError::fileNotFound };

    auto xml = juce::XmlDocument::parse (file);
    if (xml == nullptr)
        return { {}, ReadError::parseFailed };

    auto tree = juce::ValueTree::fromXml (*xml);
    if (!tree.isValid())
        return { {}, ReadError::treeConversionFailed };

    return { tree, ReadError::none };
}

//==============================================================================
// Rolling Backups
//==============================================================================

juce::String XmlPersistence::backupTimestamp()
{
    const auto now = juce::Time::getCurrentTime();
    return now.formatted ("%Y%m%d_%H%M%S") + "_" + juce::String (now.getMilliseconds()).paddedLeft ('0', 3);
}

XmlPersistence::BackupResult XmlPersistence::backUpFile (const juce::File& file, const juce::File& backupFolder)
{
    if (!file.existsAsFile())
        return { true, {} };

    // A default File would put the copy at the root of the drive.
    if (backupFolder == juce::File() || backupFolder.createDirectory().failed())
        return {};

    // Never onto an existing backup. File::copyFileTo deletes its target
    // first, and names had one-second resolution: two saves of one file in
    // the same second silently replaced the first save's backup (the only
    // copy of the version before it), and with that backup held open by a
    // sync client or an antivirus the delete failed and the save was
    // refused (re-audit 2026-09-29, R5). Milliseconds make a clash rare, and
    // a clash that still happens gets a numbered sibling, which the
    // `<prefix>_*.*` listing still finds.
    auto backupFile = backupFolder.getChildFile (
        file.getFileNameWithoutExtension() + "_" + backupTimestamp() + file.getFileExtension());
    if (backupFile.exists())
        backupFile = backupFile.getNonexistentSibling (true);

    if (! file.copyFileTo (backupFile))
        return {};

    return { true, backupFile };
}

bool XmlPersistence::createBackup (const juce::File& file, const juce::File& backupFolder)
{
    return backUpFile (file, backupFolder).ok;
}

juce::Array<juce::File> XmlPersistence::listBackups (const juce::File& backupFolder,
                                                     const juce::String& filePrefix)
{
    juce::Array<juce::File> backups;

    if (backupFolder.isDirectory())
    {
        auto files = backupFolder.findChildFiles (juce::File::findFiles, false, filePrefix + "_*.*");

        // Sort by modification time (newest first)
        std::sort (files.begin(), files.end(),
            [] (const juce::File& a, const juce::File& b)
            {
                return a.getLastModificationTime() > b.getLastModificationTime();
            });

        for (auto& file : files)
            backups.add (file);
    }

    return backups;
}

void XmlPersistence::cleanupBackups (const juce::File& backupFolder,
                                     const juce::StringArray& filePrefixes,
                                     int keepCount)
{
    for (const auto& prefix : filePrefixes)
    {
        auto backups = listBackups (backupFolder, prefix);
        for (int i = keepCount; i < backups.size(); ++i)
            backups[i].deleteFile();
    }
}

//==============================================================================
// Merge / Backfill Engine (preserves missing properties)
//==============================================================================

void XmlPersistence::mergeProperties (juce::ValueTree& target, const juce::ValueTree& source,
                                      juce::UndoManager* undoManager) const
{
    // Only copy properties that exist in source - missing properties keep their current value.
    //
    // Each candidate value passes through the injected per-property validator
    // first (the app's file-load value gate). A rejected property keeps its
    // current target value; the app owns diagnostics for rejections.
    for (int i = 0; i < source.getNumProperties(); ++i)
    {
        const auto propName = source.getPropertyName (i);
        auto v              = source.getProperty (propName);

        if (options.propertyValidator != nullptr)
        {
            auto validated = options.propertyValidator (propName, v);
            if (!validated.has_value())
                continue;
            v = std::move (*validated);
        }

        target.setProperty (propName, v, undoManager);
    }
}

void XmlPersistence::mergeTreeRecursive (juce::ValueTree& target, const juce::ValueTree& source,
                                         juce::UndoManager* undoManager) const
{
    const auto& id = options.idProperty;

    // Merge properties (only those in source)
    mergeProperties (target, source, undoManager);

    // Merge children
    for (int i = 0; i < source.getNumChildren(); ++i)
    {
        auto sourceChild = source.getChild (i);
        juce::ValueTree targetChild;

        // For children with an "id" property (channel-style collections),
        // match by both type AND id to avoid mixing up channels.
        // The match must scan for type+id together: getChildWithProperty()
        // returns the first child of ANY type with that id, and siblings of
        // different types can share an id namespace. Invalidating on a type
        // mismatch made every same-id-different-type child look "missing",
        // so each load appended duplicate nodes and the file grew on every
        // load/save cycle.
        if (sourceChild.hasProperty (id))
        {
            const auto sourceId = sourceChild.getProperty (id);
            for (int c = 0; c < target.getNumChildren(); ++c)
            {
                auto candidate = target.getChild (c);
                if (candidate.getType() == sourceChild.getType()
                    && candidate.getProperty (id) == sourceId)
                {
                    targetChild = candidate;
                    break;
                }
            }
        }
        else
        {
            // For children without an id, match by type AND ordinal position among
            // same-type, id-less siblings. Matching by type name alone returns the
            // FIRST such child for every source child, so a collection of repeated
            // id-less children (e.g. a preset list) would all collapse into the
            // first slot and the rest would be lost on load.
            const auto type = sourceChild.getType();
            int wanted = 0;
            for (int j = 0; j < i; ++j)
            {
                auto prevSource = source.getChild (j);
                if (prevSource.getType() == type && ! prevSource.hasProperty (id))
                    ++wanted;
            }

            int seen = 0;
            for (int c = 0; c < target.getNumChildren(); ++c)
            {
                auto candidate = target.getChild (c);
                if (candidate.getType() == type && ! candidate.hasProperty (id))
                {
                    if (seen == wanted) { targetChild = candidate; break; }
                    ++seen;
                }
            }
        }

        if (targetChild.isValid())
        {
            // Child exists - recursively merge
            mergeTreeRecursive (targetChild, sourceChild, undoManager);
        }
        else
        {
            // Child doesn't exist in target - add it (new section in file)
            target.appendChild (sourceChild.createCopy(), undoManager);
        }
    }
}

} // namespace spatcore::control::state
