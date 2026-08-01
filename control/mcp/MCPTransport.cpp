#include "MCPTransport.h"

namespace spatcore::control::mcp
{

namespace
{
    constexpr const char* kEndpointPath = "/mcp";

    SimpleWeb::CaseInsensitiveMultimap defaultHeaders (bool loopbackOnly)
    {
        SimpleWeb::CaseInsensitiveMultimap h;
        h.emplace ("Content-Type", "application/json");
        // Loopback-bound: a wildcard `*` is acceptable because the socket
        // itself is bound to 127.0.0.1, so only same-machine origins can
        // reach this endpoint at all. LAN-bound: restrict to "null" to
        // refuse browser CORS preflight from arbitrary LAN pages — the
        // real auth story is still TODO (no token model yet), so the
        // tightest CORS posture is the only safety we have.
        h.emplace ("Access-Control-Allow-Origin", loopbackOnly ? "*" : "null");
        h.emplace ("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
        // MCP-Protocol-Version must be listed: from spec revision 2025-06-18
        // clients send it on every post-initialize request, and a browser
        // client would otherwise fail preflight before reaching us.
        h.emplace ("Access-Control-Allow-Headers",
                   "Content-Type, Authorization, MCP-Protocol-Version");
        return h;
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

    if (method == "OPTIONS")
    {
        // CORS preflight — answer with empty body and the Allow* headers from
        // defaultHeaders(). SimpleWeb routes OPTIONS through default_resource
        // since benkuper/juce_simpleweb#5 merged.
        writeJson (response, SimpleWeb::StatusCode::success_no_content, juce::String());
        return true;
    }

    if (method == "GET")
    {
        // Streamable-HTTP server-push (SSE) lands in a later phase. For Phase 1
        // we expose request/response only, so GET is explicitly disallowed.
        writeMethodNotAllowed (response, "POST, OPTIONS");
        return true;
    }

    if (method != "POST")
    {
        writeMethodNotAllowed (response, "POST, OPTIONS");
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

            auto error = std::make_unique<juce::DynamicObject>();
            error->setProperty ("code", -32600);
            error->setProperty ("message",
                                "Unsupported MCP-Protocol-Version: "
                                + context.protocolVersionHeader
                                + ". Supported: " + protocol::supportedList());

            auto envelope = std::make_unique<juce::DynamicObject>();
            envelope->setProperty ("jsonrpc", "2.0");
            envelope->setProperty ("id", juce::var());
            envelope->setProperty ("error", juce::var (error.release()));

            writeJson (response, SimpleWeb::StatusCode::client_error_bad_request,
                       juce::JSON::toString (juce::var (envelope.release()), true));
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
        writeJson (response, SimpleWeb::StatusCode::server_error_service_unavailable, err);
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
        writeJson (response, SimpleWeb::StatusCode::server_error_internal_server_error, err);
        return true;
    }

    writeJson (response, SimpleWeb::StatusCode::success_ok, responseBody);
    return true;
}

void MCPTransport::writeJson (std::shared_ptr<HttpServer::Response> response,
                              SimpleWeb::StatusCode statusCode,
                              const juce::String& body,
                              const SimpleWeb::CaseInsensitiveMultimap& extraHeaders) const
{
    auto headers = defaultHeaders (loopbackOnlyMode);
    for (const auto& kv : extraHeaders)
        headers.emplace (kv.first, kv.second);

    response->write (statusCode, body.toStdString(), headers);
}

void MCPTransport::writeMethodNotAllowed (std::shared_ptr<HttpServer::Response> response,
                                          const juce::String& allowedMethods) const
{
    SimpleWeb::CaseInsensitiveMultimap h;
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
