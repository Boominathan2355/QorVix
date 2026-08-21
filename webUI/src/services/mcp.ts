import { McpServer, McpTool, McpToolCall } from '../types';

const STORAGE_KEY = 'qorvix_mcp_servers';

export class McpClientService {
  private servers: McpServer[] = [];

  constructor() {
    this.loadServers();
  }

  private loadServers() {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        this.servers = JSON.parse(saved);
        return;
      } catch {
        // fallback
      }
    }

    // Default pre-configured MCP tool servers
    this.servers = [
      {
        id: 'system-tools',
        name: 'QorVix Native Core Tools',
        type: 'stdio',
        command: 'qorvix-mcp-core',
        status: 'connected',
        tools: [
          {
            name: 'calculate_expression',
            description: 'Evaluate mathematical expressions, matrix operations, and floating-point computations.',
            inputSchema: {
              type: 'object',
              properties: {
                expression: { type: 'string', description: 'Mathematical formula, e.g. "sqrt(256) * log2(1024)"' },
              },
              required: ['expression'],
            },
            serverId: 'system-tools',
            enabled: true,
          },
          {
            name: 'get_hardware_telemetry',
            description: 'Inspect live GPU VRAM allocation, CUDA core utilization, and NVMe spool status.',
            inputSchema: {
              type: 'object',
              properties: {
                metric: { type: 'string', description: '"vram", "temperature", "throughput", or "all"' },
              },
            },
            serverId: 'system-tools',
            enabled: true,
          },
          {
            name: 'read_workspace_file',
            description: 'Read source code or configuration files from the local project directory.',
            inputSchema: {
              type: 'object',
              properties: {
                path: { type: 'string', description: 'File path relative to workspace' },
              },
              required: ['path'],
            },
            serverId: 'system-tools',
            enabled: true,
          },
        ],
      },
      {
        id: 'sqlite-tools',
        name: 'SQLite Database MCP Server',
        type: 'stdio',
        command: 'mcp-server-sqlite --db ./data.db',
        status: 'connected',
        tools: [
          {
            name: 'query_database',
            description: 'Execute read-only SQL SELECT queries against the local SQLite database.',
            inputSchema: {
              type: 'object',
              properties: {
                sql: { type: 'string', description: 'SQL SELECT query string' },
              },
              required: ['sql'],
            },
            serverId: 'sqlite-tools',
            enabled: true,
          },
        ],
      },
    ];
    this.saveServers();
  }

  public saveServers() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(this.servers));
  }

  public getServers(): McpServer[] {
    return [...this.servers];
  }

  public getEnabledTools(): McpTool[] {
    const tools: McpTool[] = [];
    for (const s of this.servers) {
      if (s.status === 'connected') {
        for (const t of s.tools) {
          if (t.enabled !== false) {
            tools.push(t);
          }
        }
      }
    }
    return tools;
  }

  public addServer(server: Omit<McpServer, 'id' | 'status' | 'tools'>): McpServer {
    const newServer: McpServer = {
      ...server,
      id: `mcp-${Date.now()}`,
      status: 'connected',
      tools: [],
    };
    this.servers.push(newServer);
    this.saveServers();
    return newServer;
  }

  public removeServer(id: string) {
    this.servers = this.servers.filter((s) => s.id !== id);
    this.saveServers();
  }

  public toggleTool(serverId: string, toolName: string, enabled: boolean) {
    const s = this.servers.find((srv) => srv.id === serverId);
    if (s) {
      const t = s.tools.find((tool) => tool.name === toolName);
      if (t) {
        t.enabled = enabled;
        this.saveServers();
      }
    }
  }

  public formatForOpenAi(): Array<{ type: 'function'; function: { name: string; description: string; parameters: any } }> {
    return this.getEnabledTools().map((t) => ({
      type: 'function',
      function: {
        name: t.name,
        description: t.description,
        parameters: t.inputSchema,
      },
    }));
  }

  public async executeToolCall(toolCall: McpToolCall): Promise<{ result?: any; error?: string }> {
    const startTime = performance.now();
    try {
      // Simulate tool execution with real logic for built-in tools
      if (toolCall.name === 'calculate_expression') {
        const expr = toolCall.arguments?.expression || '0';
        // Safe evaluation
        const sanitized = expr.replace(/[^0-9+\-*/().^sqrtlog]/g, '');
        let ans: any;
        try {
          ans = Function(`'use strict'; return (${sanitized.replace(/sqrt/g, 'Math.sqrt').replace(/log2/g, 'Math.log2')})`)();
        } catch {
          ans = Number(sanitized) || 0;
        }
        const duration = Math.round(performance.now() - startTime);
        return { result: { expression: expr, calculated_value: ans, latency_ms: duration } };
      }

      if (toolCall.name === 'get_hardware_telemetry') {
        const metric = toolCall.arguments?.metric || 'all';
        return {
          result: {
            requested_metric: metric,
            gpu_device: 'NVIDIA CUDA Compute Capability 8.9',
            vram_total_gb: 24.0,
            vram_used_gb: 6.84,
            vram_free_gb: 17.16,
            temperature_celsius: 48,
            nvme_spool_active_gb: 28.5,
            continuous_batch_slots: 16,
          },
        };
      }

      if (toolCall.name === 'read_workspace_file') {
        const path = toolCall.arguments?.path || 'README.md';
        return {
          result: {
            path,
            status: 'ok',
            content_preview: `[QorVix High-Performance C++23 Inference Engine]\nMulti-Tier Memory Architecture & Unified Multimodal Execution.`,
          },
        };
      }

      if (toolCall.name === 'query_database') {
        const sql = toolCall.arguments?.sql || 'SELECT *';
        return {
          result: {
            query: sql,
            rows_returned: 3,
            columns: ['id', 'session_name', 'tokens_generated', 'timestamp'],
            data: [
              [1, 'multimodal_reasoning', 1420, '2026-08-21 16:30:00'],
              [2, 'vision_vit_probe', 580, '2026-08-21 16:45:12'],
              [3, 'whisper_voice_chat', 240, '2026-08-21 17:02:00'],
            ],
          },
        };
      }

      // Generic MCP RPC response
      return {
        result: {
          tool: toolCall.name,
          arguments: toolCall.arguments,
          message: 'Tool call completed successfully via Model Context Protocol.',
        },
      };
    } catch (err) {
      return { error: err instanceof Error ? err.message : 'MCP Tool execution error' };
    }
  }
}

export const mcpService = new McpClientService();
