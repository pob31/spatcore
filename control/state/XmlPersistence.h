#pragma once

#include <juce_core/juce_core.h>
#include <juce_data_structures/juce_data_structures.h>
#include <functional>
#include <optional>

namespace spatcore::control::state
{

/**
 * XmlPersistence — app-agnostic ValueTree↔XML persistence machinery.
 *
 * Extracted from the app's WFSFileManager (Phase 4c-3 of the spatcore
 * extraction). Owns the mechanics that carry no app schema:
 *
 *  - XML file write with the commented header convention
 *    (`<!-- <title> --> / <!-- Type: ... --> / <!-- Created: ... -->`).
 *    The title line is app data, injected at construction.
 *  - generic XML file read/parse into a ValueTree, reporting the failure
 *    stage (missing file / parse error / tree conversion) so the app can
 *    map it to its own (localized) error strings.
 *  - timestamped rolling backups: copy-aside into a backups folder with a
 *    `<name>_<yyyymmdd_hhmmss>` suffix, newest-first listing by prefix,
 *    and keep-last-N retention.
 *  - the merge/backfill engine (mergeProperties / mergeTreeRecursive):
 *    "missing = keep" property merge plus recursive child matching by
 *    type + id (and by type + ordinal among id-less siblings). Per-property
 *    value validation is app policy and is injected as a std::function —
 *    the core never names a parameter or a bounds table.
 *
 * Everything schema-shaped — the section-split file layout, manifest
 * handling, snapshots, migrations, dialogs — stays in the app.
 */
class XmlPersistence
{
public:
    //==========================================================================
    // Configuration
    //==========================================================================

    /** Per-property merge validator. Called for every property about to be
     *  merged into the target tree. Return the value to apply (usually the
     *  input value, unchanged), or std::nullopt to reject the property — the
     *  target then keeps its current value. The app owns logging/diagnostics
     *  for rejected values. A null function accepts everything. */
    using PropertyValidator =
        std::function<std::optional<juce::var> (const juce::Identifier& property,
                                                const juce::var& value)>;

    struct Options
    {
        /** First comment line written into every XML file header. */
        juce::String headerTitle;

        /** Property used to match children across trees during merges
         *  (channel-style children carry a stable id). */
        juce::Identifier idProperty { "id" };

        /** Optional per-property merge validator (see PropertyValidator). */
        PropertyValidator propertyValidator;
    };

    explicit XmlPersistence (Options options);

    //==========================================================================
    // XML File I/O
    //==========================================================================

    enum class WriteResult
    {
        ok,
        xmlConversionFailed,   ///< ValueTree could not be converted to XML
        fileWriteFailed        ///< File could not be written
    };

    enum class ReadError
    {
        none,
        fileNotFound,          ///< File does not exist
        parseFailed,           ///< File exists but is not parseable XML
        treeConversionFailed   ///< XML parsed but produced no valid ValueTree
    };

    struct ReadResult
    {
        juce::ValueTree tree;
        ReadError error = ReadError::none;
    };

    /** Write a ValueTree to an XML file with the commented header convention
     *  (human-readable, no JUCE XML declaration of its own — the header
     *  carries it). The header's Type line is the file's base name. Lines
     *  end in CRLF on every platform, as they always have. Either the whole
     *  file is replaced or it is left as it was (see replaceFileContents). */
    WriteResult writeTreeToFile (const juce::ValueTree& tree, const juce::File& file) const;

    /** Replace `file` with exactly `numBytes` bytes, or leave it as it was.
     *  The bytes go to a hidden temporary file beside it, which is flushed
     *  to the disk, checked for its full length and only then renamed over
     *  the file; on any failure the temporary file is deleted and false is
     *  returned. juce::File::replaceWithText takes the same steps but never
     *  checks the write, so a full disk or a pulled drive swapped the file
     *  for a truncated one and still reported success. */
    static bool replaceFileContents (const juce::File& file, const void* data, size_t numBytes);

    /** Read a ValueTree from an XML file, reporting the failure stage. */
    ReadResult readTreeFromFile (const juce::File& file) const;

    /** Build the commented XML header for a given file type. */
    juce::String makeXmlHeader (const juce::String& fileType) const;

    //==========================================================================
    // Rolling Backups
    //==========================================================================

    /** Timestamp suffix used for backup file names (yyyymmdd_hhmmss_mmm). */
    static juce::String backupTimestamp();

    /** What a backup did: `ok` is false when the file exists and no copy
     *  could be made; `copy` is the copy made, or a default File when the
     *  source did not exist (nothing to back up, and `ok`). */
    struct BackupResult
    {
        bool ok = false;
        juce::File copy;
    };

    /** Copy a file aside into the backup folder as
     *  `<name>_<timestamp><ext>`, never onto an existing backup: a name
     *  already taken gets a numbered sibling. Creates the backup folder if
     *  needed. A caller about to overwrite the file should not go ahead when
     *  `ok` is false: the version it would lose has no copy. */
    static BackupResult backUpFile (const juce::File& file, const juce::File& backupFolder);

    /** backUpFile, reduced to its `ok`. */
    static bool createBackup (const juce::File& file, const juce::File& backupFolder);

    /** List backups for a file-name prefix, newest first. */
    static juce::Array<juce::File> listBackups (const juce::File& backupFolder,
                                                const juce::String& filePrefix);

    /** Delete all but the newest keepCount backups for each prefix. */
    static void cleanupBackups (const juce::File& backupFolder,
                                const juce::StringArray& filePrefixes,
                                int keepCount);

    //==========================================================================
    // Merge / Backfill Engine
    //==========================================================================

    /** Merge properties from source into target. Only properties present in
     *  source are copied — missing properties keep their current value.
     *  Each property passes through the injected validator first; rejected
     *  properties keep the target's current value. */
    void mergeProperties (juce::ValueTree& target, const juce::ValueTree& source,
                          juce::UndoManager* undoManager) const;

    /** Recursively merge source into target, preserving target properties
     *  and children absent from source. Children carrying the id property
     *  are matched by type AND id; id-less children are matched by type and
     *  ordinal position among same-type id-less siblings. Unmatched source
     *  children are appended as copies. */
    void mergeTreeRecursive (juce::ValueTree& target, const juce::ValueTree& source,
                             juce::UndoManager* undoManager) const;

private:
    Options options;
};

} // namespace spatcore::control::state
