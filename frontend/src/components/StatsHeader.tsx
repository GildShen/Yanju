import { useState, useEffect } from 'react';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { api, type Stats } from '@/lib/api';

export function StatsHeader() {
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    api<Stats>('/api/stats')
      .then(setStats)
      .catch(() => {});
  }, []);

  if (!stats) return null;

  return (
    <div className="flex items-center gap-3 text-sm">
      <Badge variant="secondary" className="gap-1">
        📄 <span className="font-semibold">{stats.papers}</span> papers
      </Badge>
      <Separator orientation="vertical" className="h-4" />
      <Badge variant="secondary" className="gap-1">
        🧬 <span className="font-semibold">{stats.embeddings}</span> embeddings
      </Badge>
      <Separator orientation="vertical" className="h-4" />
      <Badge variant="secondary" className="gap-1">
        📁 <span className="font-semibold">{stats.pdf_catalog}</span> PDFs
      </Badge>
      <Separator orientation="vertical" className="h-4" />
      <Badge variant="secondary" className="gap-1">
        🔬 <span className="font-semibold">{stats.methodology_runs}</span> analyses
      </Badge>
    </div>
  );
}
