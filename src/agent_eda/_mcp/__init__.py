"""The MCP adapter: everything between an MCP client and the analyses.

`agent_eda.mcp` is the public entry point and the only module outside this
package that imports it. Of the modules here, only the one that builds the MCP
server imports mcp itself, so `agent_eda.mcp` can check that the extra is
installed first. Nothing else in the package imports from here.
"""
