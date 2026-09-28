#include "MCPTransport.h"
#include "MCPRequestGuards.h"
#include "../osc/NetworkStringUtils.h"

namespace spatcore::control::mcp
{

namespace
{
    constexpr const char* kEndpointPath = "/mcp";

    SimpleWeb::CaseInsensitiveMultimap defaultHeaders()
    {
        SimpleWeb::CaseInsensitiveMultimap h;
        h.emplace ("Content-Type", "application/json");
        return h;
    }

    /** The CORS answer for a page on a loopback origin (the only origin let
        through): that origin by name, never `*`, which would let any page
        read the replies. */
    SimpleWeb::CaseInsensitiveMultimap corsHeaders (const juce::String& origin)
    {
        SimpleWeb::CaseInsensitiveMultimap h;
        h.emplace ("Access-Control-Allow-Origin", origin.toStdString());
        h.emplace ("Vary", "Origin");
        h.emplace ("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
        // MCP-Protocol-Version must be listed: from spec revision 2025-06-18
        // clients send it on every post-initialize request, and a browser
        // client would otherwise fail preflight before reaching us.
        h.emplace ("Access-Control-Allow-Headers",
                   "Content-Type, Authorization, MCP-Protocol-Version");
        return h;
    }

    /** A header's value read as UTF-8 and trimmed; empty when it is absent
        or is not UTF-8. */
    juce::String headerValue (const SimpleWeb::CaseInsensitiveMultimap& headers, const char* name)
    {
        const auto it = headers.find (name);
        if (it == headers.end())
            return {};

        return osc::safeStringFromBytes (it->second.data(), static_cast<int> (it->second.size())).trim();
    }

    /** A JSON-RPC error envelope with no id, for a request refused before
        its body was read. */
    juce::String refusalBody (int code, const juce::String& message)
    {
        auto error = std::make_unique<juce::DynamicObject>();
        error->setProperty ("code", code);
        error->setProperty ("message", message);

        auto envelope = std::make_unique<juce::DynamicObject>();
        envelope->setProperty ("jsonrpc", "2.0");
        envelope->setProperty ("id", juce::var());
        envelope->setProperty ("error", juce::var (error.release()));

        return juce::JSON::toString (juce::var (envelope.release()), true);
    }

    constexpr const char* kProtocolVersionHeader = "MCP-Protocol-Version";

    /** Probe whether `port` can be bound before handing it to SimpleWeb.

        SimpleWeb reports bind failures only by calling
        Listener::serverInitError on its own server thread. Taking that
        route previously crashed at app teardown: `webSocketListeners` is a
        plain juce::ListenerList, so removing a listener from the message
        thread while the server thread is inside .call() is a data race, and
        the juce::String argument is constructed on the server thread from a
        listener that may already be gone. Rather than re-introduce that
        hazard for a callback that fires exactly once, we ask the OS the
        same question up front, on the calling thread, with no shared state.

        The trade-off is honest: this is a probe, not a confirmation, so a
        port stolen in the microseconds between probe and real bind would
        still slip through. It catches the failure that actually happens
        (another process already listening), which is all the UI needs to
        stop claiming the server is up when it isn't.

        One platform nuance worth knowing before trusting this too far:
        `createListener` sets SO_REUSEADDR everywhere except Windows
        (juce_Socket.cpp, guarded by `#if ! JUCE_WINDOWS`), while SimpleWeb
        binds with allowAddressReuse=false. An active listener fails the
        bind either way, on every platform — that is the case we care about.
        But on macOS/Linux a port held only in TIME_WAIT can pass this probe
        and then fail SimpleWeb's stricter bind, which lands back on the old
        silent-failure behaviour. Not a regression, just not a full fix
        there. */
    bool canBindPort (int port, bool loopbackOnly)
    {
        if (port <= 0)
            return true;  // 0 means "let the OS choose" — nothing to probe.

        juce::StreamingSocket probe;
        const juce::String localAddress = loopbackOnly ? juce::String ("127.0.0.1")
                                                       : juce::String();
        if (! probe.createListener (port, localAddress))
            return false;

        probe.close();
        return true;
    }
}

MCPTransport::MCPTransport (MCPLogSink& l) : mcpLogger (l) {}

MCPTransport::~MCPTransport()
{
    stop();
}

bool MCPTransport::start (int port, bool loopbackOnly)
{
    if (running.load())
        stop();

    // Verify the port is free before spawning the server thread. Without
    // this the call below always "succeeds" — SimpleWeb swallows the bind
    // error on its own thread — and the UI shows a listening server that
    // never accepted a connection.
    if (! canBindPort (port, loopbackOnly))
    {
        boundPort = 0;
        running = false;
        mcpLogger.logError ("MCP server failed to start: port " + juce::String (port)
                            + " is already in use. MCP clients will not be able to connect.");
        return false;
    }

    server = std::make_unique<SimpleWebSocketServer>();
    server->addHTTPRequestHandler (this);

    const juce::String localAddress = loopbackOnly ? juce::String ("127.0.0.1") : juce::String();

    server->start (port, /*wsSuffix*/ "", localAddress, /*allowAddressReuse*/ false);

    boundPort = port;
    loopbackOnlyMode = loopbackOnly;
    running = true;

    mcpLogger.logInfo ("MCP server listening on "
                       + (loopbackOnly ? juce::String ("127.0.0.1:") : juce::String ("0.0.0.0:"))
                       + juce::String (port) + kEndpointPath);
    return true;
}

void MCPTransport::stop()
{
    if (! running.load())
        return;

    running = false;

    if (server != nullptr)
    {
        server->removeHTTPRequestHandler (this);
        server->stop();
        server.reset();
    }

    boundPort = 0;
    mcpLogger.logInfo ("MCP server stopped");
}

void MCPTransport::setRequestHandler (HandlerCallback cb)
{
    const juce::ScopedLock sl (handlerLock);
    handler = std::move (cb);
}

bool MCPTransport::handleHTTPRequest (std::shared_ptr<HttpServer::Response> response,
                                       std::shared_ptr<HttpServer::Request> request)
{
    juce::String path   = juce::String (request->path);
    juce::String method = juce::String (request->method);

    // Normalize trailing slash so "/mcp" and "/mcp/" both match.
    if (path.length() > 1 && path.endsWithChar ('/'))
        path = path.dropLastCharacters (1);

    if (path != kEndpointPath)
    {
        // Let other handlers (or the default 404) take care of unknown paths.
        return false;
    }

    // Whatever the method: refuse what only a browser sends, before anything
    // else runs (see MCPRequestGuards.h). A second Host or Origin header is
    // refused too, so no check reads one value while a later reader sees
    // another.
    const auto& headers = request->header;
    const juce::String host = headerValue (headers, "Host");
    if (headers.count ("Host") != 1 || ! guards::isAllowedHost (host, loopbackOnlyMode))
    {
        mcpLogger.logError ("Refused a request addressed to Host \"" + host.substring (0, 80)
                            + "\": the MCP server answers only " + (loopbackOnlyMode
                                  ? juce::String ("127.0.0.1, localhost and [::1]")
                                  : juce::String ("localhost and IP addresses")));
        writeJson (response, SimpleWeb::StatusCode::client_error_forbidden,
                   refusalBody (-32600, "Host not allowed: " + host.substring (0, 80)));
        return true;
    }

    SimpleWeb::CaseInsensitiveMultimap cors;
    if (headers.count ("Origin") != 0)
    {
        const juce::String origin = headerValue (headers, "Origin");
        if (headers.count ("Origin") != 1 || ! guards::isAllowedOrigin (origin))
        {
            mcpLogger.logError ("Refused a request from the web page origin \""
                                + origin.substring (0, 80)
                                + "\": only AI clients on this machine may use the MCP server");
            writeJson (response, SimpleWeb::StatusCode::client_error_forbidden,
                       refusalBody (-32600, "Origin not allowed: " + origin.substring (0, 80)));
            return true;
        }

        cors = corsHeaders (origin);
    }

    if (method == "OPTIONS")
    {
        // CORS preflight — answer with empty body, and with the Allow*
        // headers only for a loopback origin. SimpleWeb routes OPTIONS
        // through default_resource since benkuper/juce_simpleweb#5 merged.
        writeJson (response, SimpleWeb::StatusCode::success_no_content, juce::String(), cors);
        return true;
    }

    if (method == "GET")
    {
        // Streamable-HTTP server-push (SSE) lands in a later phase. For Phase 1
        // we expose request/response only, so GET is explicitly disallowed.
        writeMethodNotAllowed (response, "POST, OPTIONS", cors);
        return true;
    }

    if (method != "POST")
    {
        writeMethodNotAllowed (response, "POST, OPTIONS", cors);
        return true;
    }

    // A page can POST text/plain, form data or multipart with no preflight;
    // application/json it cannot. Every MCP client sends application/json.
    const juce::String contentType = headerValue (headers, "Content-Type");
    if (headers.count ("Content-Type") != 1 || ! guards::isJsonContentType (contentType))
    {
        mcpLogger.logError ("Refused a request with Content-Type \"" + contentType.substring (0, 80)
                            + "\": MCP requests are application/json");
        writeJson (response, SimpleWeb::StatusCode::client_error_unsupported_media_type,
                   refusalBody (-32600, "Content-Type must be application/json"), cors);
        return true;
    }

    // POST /mcp — read body, hand to dispatcher, return its JSON-RPC envelope.
    RequestContext context;
    context.clientIP   = resolveClientIP (request);
    context.clientPort = resolveClientPort (request);

    // Spec revision 2025-06-18 onwards: clients send MCP-Protocol-Version on
    // every request after initialize. Reject an unsupported value here, with
    // HTTP 400 as the spec prescribes, so the dispatcher only ever sees
    // revisions it can honour. An absent header is fine — that means an
    // older client, and RequestContext supplies the mandated fallback.
    if (auto it = request->header.find (kProtocolVersionHeader); it != request->header.end())
    {
        context.protocolVersionHeader = juce::String (it->second).trim();

        if (context.protocolVersionHeader.isNotEmpty()
            && ! protocol::isSupported (context.protocolVersionHeader))
        {
            mcpLogger.logError ("Rejected request with unsupported "
                                + juce::String (kProtocolVersionHeader) + ": "
                                + context.protocolVersionHeader);

            writeJson (response, SimpleWeb::StatusCode::client_error_bad_request,
                       refusalBody (-32600, "Unsupported MCP-Protocol-Version: "
                                                + context.protocolVersionHeader
                                                + ". Supported: " + protocol::supportedList()),
                       cors);
            return true;
        }
    }

    juce::String body = juce::String (request->content.string());

    HandlerCallback cb;
    {
        const juce::ScopedLock sl (handlerLock);
        cb = handler;
    }

    if (! cb)
    {
        // Server is up but the dispatcher hasn't been wired yet (Phase 1 Block 3
        // can hit this path during integration). Return a structured 503 so the
        // client knows to retry later rather than treating it as a hard failure.
        const juce::String err =
            R"({"jsonrpc":"2.0","id":null,"error":{"code":-32603,)"
            R"("message":"MCP dispatcher not initialized"}})";
        writeJson (response, SimpleWeb::StatusCode::server_error_service_unavailable, err, cors);
        return true;
    }

    juce::String responseBody;
    try
    {
        responseBody = cb (body, context);
    }
    catch (const std::exception& e)
    {
        mcpLogger.logError (juce::String ("Dispatcher threw: ") + e.what());
        const juce::String err =
            R"({"jsonrpc":"2.0","id":null,"error":{"code":-32603,)"
            R"("message":"Internal server error"}})";
        writeJson (response, SimpleWeb::StatusCode::server_error_internal_server_error, err, cors);
        return true;
    }

    writeJson (response, SimpleWeb::StatusCode::success_ok, responseBody, cors);
    return true;
}

void MCPTransport::writeJson (std::shared_ptr<HttpServer::Response> response,
                              SimpleWeb::StatusCode statusCode,
                              const juce::String& body,
                              const SimpleWeb::CaseInsensitiveMultimap& extraHeaders) const
{
    auto headers = defaultHeaders();
    for (const auto& kv : extraHeaders)
        headers.emplace (kv.first, kv.second);

    response->write (statusCode, body.toStdString(), headers);
}

void MCPTransport::writeMethodNotAllowed (std::shared_ptr<HttpServer::Response> response,
                                          const juce::String& allowedMethods,
                                          const SimpleWeb::CaseInsensitiveMultimap& cors) const
{
    SimpleWeb::CaseInsensitiveMultimap h (cors);
    h.emplace ("Allow", allowedMethods.toStdString());
    const juce::String body = R"({"error":"method_not_allowed"})";
    writeJson (response, SimpleWeb::StatusCode::client_error_method_not_allowed, body, h);
}

juce::String MCPTransport::resolveClientIP (const std::shared_ptr<HttpServer::Request>& request)
{
    try
    {
        auto endpoint = request->remote_endpoint();
        return juce::String (endpoint.address().to_string());
    }
    catch (...)
    {
        return juce::String();
    }
}

int MCPTransport::resolveClientPort (const std::shared_ptr<HttpServer::Request>& request)
{
    try
    {
        return static_cast<int> (request->remote_endpoint().port());
    }
    catch (...)
    {
        return 0;
    }
}

} // namespace spatcore::control::mcp
