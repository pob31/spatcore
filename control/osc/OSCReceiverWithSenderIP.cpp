// The system's socket headers first, before JUCE's, for the one socket
// option JUCE does not set: the receive buffer's size.
#if defined (_WIN32)
 #ifndef NOMINMAX
  #define NOMINMAX
 #endif
 #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
 #endif
 #include <winsock2.h>
#else
 #include <sys/socket.h>
#endif

#include "OSCReceiverWithSenderIP.h"
#include "OSCParser.h"

namespace spatcore::control::osc
{

namespace
{
    // Asks for `bytes` of receive buffer and answers what the socket has
    // afterwards; the system may give less (Linux caps it at rmem_max).
    int askForReceiveBuffer (int handle, int bytes)
    {
        if (handle < 0)
            return 0;

       #if defined (_WIN32)
        const auto socketHandle = static_cast<SOCKET> (handle);
        using Length = int;
       #else
        const auto socketHandle = handle;
        using Length = socklen_t;
       #endif

        (void) setsockopt (socketHandle, SOL_SOCKET, SO_RCVBUF,
                           reinterpret_cast<const char*> (&bytes), static_cast<Length> (sizeof (bytes)));

        int granted = 0;
        auto length = static_cast<Length> (sizeof (granted));

        if (getsockopt (socketHandle, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<char*> (&granted), &length) != 0)
            return 0;

        return granted;
    }
}

//==============================================================================
// OSCReceiverWithSenderIP
//==============================================================================

OSCReceiverWithSenderIP::OSCReceiverWithSenderIP()
    : Thread("OSCReceiverWithSenderIP")
{
}

OSCReceiverWithSenderIP::~OSCReceiverWithSenderIP()
{
    disconnect();
}

bool OSCReceiverWithSenderIP::connect(int port)
{
    if (connected.load())
        disconnect();

    socket = std::make_unique<juce::DatagramSocket>();

    if (!socket->bindToPort(port))
    {
        DBG("OSCReceiverWithSenderIP: Failed to bind to port " << port);
        socket.reset();
        return false;
    }

    // Room for a burst, before anything arrives.
    receiveBufferBytes = askForReceiveBuffer (socket->getRawSocketHandle(), kReceiveBufferWanted);

    portNumber = port;
    shouldStop = false;
    connected = true;
    startThread();

    return true;
}

bool OSCReceiverWithSenderIP::disconnect()
{
    if (!connected.load())
        return true;

    shouldStop = true;

    if (socket)
        socket->shutdown();

    stopThread(2000);

    socket.reset();
    connected = false;
    portNumber = 0;
    receiveBufferBytes = 0;

    return true;
}

void OSCReceiverWithSenderIP::addListener(Listener* listener)
{
    listeners.add(listener);
}

void OSCReceiverWithSenderIP::removeListener(Listener* listener)
{
    listeners.remove(listener);
}

void OSCReceiverWithSenderIP::run()
{
    juce::HeapBlock<char> buffer(bufferSize);

    while (!threadShouldExit() && !shouldStop.load())
    {
        // Wait for data with timeout (100ms) to allow thread exit checking
        int ready = socket->waitUntilReady(true, 100);

        if (ready < 0)
        {
            // Error occurred
            DBG("OSCReceiverWithSenderIP: Socket error");
            break;
        }

        if (ready == 0)
        {
            // Timeout - continue waiting
            continue;
        }

        // Data is available
        juce::String senderIP;
        int senderPort;

        int bytesRead = socket->read(buffer.getData(),
                                     bufferSize,
                                     false,
                                     senderIP,
                                     senderPort);

        if (bytesRead > 0)
        {
            // Create a copy of the data for async processing
            juce::MemoryBlock data(buffer.getData(), static_cast<size_t>(bytesRead));

            if (rawDataCallback)
            {
                // Hand raw bytes to the ingest queue (coalesce + bounded
                // FIFO + single drain callAsync). Skips the per-packet
                // parse-on-MM-thread step entirely.
                rawDataCallback(std::move(data), senderIP, senderPort);
            }
            else
            {
                // Legacy path: post to message thread for parsing and
                // listener notification, one callAsync per datagram.
                juce::MessageManager::callAsync([this, data, senderIP]()
                {
                    parseOSCData(data, senderIP);
                });
            }
        }
    }
}

void OSCReceiverWithSenderIP::setRawDataCallback(RawDataCallback callback)
{
    rawDataCallback = std::move(callback);
}

void OSCReceiverWithSenderIP::parseOSCData(const juce::MemoryBlock& data,
                                           const juce::String& senderIP)
{
    try
    {
        const char* dataPtr = static_cast<const char*>(data.getData());
        int dataSize = static_cast<int>(data.getSize());
        int pos = 0;

        // Check if this is a bundle (starts with "#bundle")
        if (dataSize >= 8 && std::memcmp(dataPtr, "#bundle", 7) == 0)
        {
            auto bundle = OSCParser::parseBundle(dataPtr, dataSize, pos);
            notifyBundle(bundle, senderIP);
        }
        else
        {
            // Try to parse as a single message
            auto message = OSCParser::parseMessage(dataPtr, dataSize, pos);
            notifyMessage(message, senderIP);
        }
    }
    catch (const juce::OSCFormatError&)
    {
        DBG("OSCReceiverWithSenderIP: Parse error from " << senderIP);
    }
}

void OSCReceiverWithSenderIP::notifyMessage(const juce::OSCMessage& message,
                                            const juce::String& senderIP)
{
    listeners.call([&](Listener& l) { l.oscMessageReceived(message, senderIP); });
}

void OSCReceiverWithSenderIP::notifyBundle(const juce::OSCBundle& bundle,
                                           const juce::String& senderIP)
{
    listeners.call([&](Listener& l) { l.oscBundleReceived(bundle, senderIP); });
}

} // namespace spatcore::control::osc
