// Prometheus metrics scraper and parser for Port 2009 /metrics

export interface ParsedMetric {
  name: string;
  help?: string;
  type?: 'gauge' | 'counter' | 'histogram' | 'summary';
  values: {
    labels: Record<string, string>;
    value: number;
  }[];
}

export function parsePrometheusMetrics(raw: string): ParsedMetric[] {
  const lines = raw.split('\n');
  const metricMap = new Map<string, ParsedMetric>();

  let currentHelp = '';
  let currentType: ParsedMetric['type'] = undefined;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    if (trimmed.startsWith('# HELP ')) {
      const parts = trimmed.slice(7).split(' ');
      const name = parts[0];
      currentHelp = parts.slice(1).join(' ');
      if (!metricMap.has(name)) {
        metricMap.set(name, { name, help: currentHelp, type: currentType, values: [] });
      } else {
        const m = metricMap.get(name)!;
        m.help = currentHelp;
      }
      continue;
    }

    if (trimmed.startsWith('# TYPE ')) {
      const parts = trimmed.slice(7).split(' ');
      const name = parts[0];
      currentType = parts[1] as ParsedMetric['type'];
      if (!metricMap.has(name)) {
        metricMap.set(name, { name, help: currentHelp, type: currentType, values: [] });
      } else {
        const m = metricMap.get(name)!;
        m.type = currentType;
      }
      continue;
    }

    if (trimmed.startsWith('#')) continue;

    // Parse metric value line: metric_name{label="val"} 123.45
    const match = trimmed.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]+)\})?\s+([0-9eE.+-]+)(?:\s+[0-9]+)?$/);
    if (match) {
      const name = match[1];
      const labelsRaw = match[2];
      const val = parseFloat(match[3]);

      const labels: Record<string, string> = {};
      if (labelsRaw) {
        const labelPairs = labelsRaw.split(',');
        for (const pair of labelPairs) {
          const [k, v] = pair.split('=');
          if (k && v) {
            labels[k.trim()] = v.trim().replace(/^"|"$/g, '');
          }
        }
      }

      if (!metricMap.has(name)) {
        metricMap.set(name, { name, help: currentHelp, type: currentType, values: [] });
      }
      metricMap.get(name)!.values.push({ labels, value: val });
    }
  }

  return Array.from(metricMap.values());
}
