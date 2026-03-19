import { useState, useEffect, useCallback } from 'react';
import { Bookmark, FileText, ExternalLink, FileArchive, Search } from 'lucide-react';
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select';
import { api, type Paper } from '@/lib/api';
import { AnalysisDialog } from '@/components/AnalysisDialog';

export function PaperExplorer() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [allPapers, setAllPapers] = useState<Paper[]>([]);
  const [search, setSearch] = useState('');
  const [dateFilter, setDateFilter] = useState('');
  const [journalFilter, setJournalFilter] = useState('ALL');
  const [dates, setDates] = useState<string[]>([]);
  const [journals, setJournals] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [analyzeEntry, setAnalyzeEntry] = useState<string | null>(null);

  // Load available dates
  useEffect(() => {
    api<{ dates: string[] }>('/api/papers/dates')
      .then(d => setDates(d.dates || []))
      .catch(() => {});
  }, []);

  const loadPapers = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ limit: '50' });
      if (search) params.set('text', search);
      if (dateFilter) params.set('published', dateFilter);
      const data = await api<Paper[]>(`/api/papers?${params}`);
      setAllPapers(data);

      // Extract unique journals
      const uniqueJournals = [...new Set(data.map(p => p.source).filter(Boolean))].sort() as string[];
      setJournals(uniqueJournals);
      setJournalFilter('ALL');

      setPapers(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [search, dateFilter]);

  useEffect(() => { loadPapers(); }, [loadPapers]);

  // Apply journal filter
  useEffect(() => {
    if (journalFilter === 'ALL') {
      setPapers(allPapers);
    } else {
      setPapers(allPapers.filter(p => p.source === journalFilter));
    }
  }, [journalFilter, allPapers]);

  const toggleStar = async (entryId: string) => {
    try {
      await api('/api/papers/toggle-star', {
        method: 'POST',
        body: JSON.stringify({ entry_id: entryId }),
      });
      setAllPapers(prev =>
        prev.map(p => {
          if (p.entry_id !== entryId) return p;
          const tags = p.tags || [];
          const starred = tags.includes('starred');
          return { ...p, tags: starred ? tags.filter(t => t !== 'starred') : [...tags, 'starred'] };
        })
      );
    } catch {}
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-[1.28rem]">Paper Explorer</CardTitle>
        <CardDescription className="text-base">Browse, search, and manage your research papers.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-3">
          <Input
            placeholder="Search papers..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="flex-1 min-w-[200px]"
          />
          <Select value={dateFilter || '__all__'} onValueChange={v => setDateFilter(v === '__all__' || !v ? '' : v)}>
            <SelectTrigger className="w-44">
              <SelectValue placeholder="All dates" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__all__">All dates</SelectItem>
              {dates.map(d => (
                <SelectItem key={d} value={d}>{d}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          {journals.length > 0 && (
            <Select value={journalFilter} onValueChange={v => setJournalFilter(v || 'ALL')}>
              <SelectTrigger className="w-52">
                <SelectValue placeholder="All journals" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="ALL">ALL</SelectItem>
                {journals.map(j => (
                  <SelectItem key={j} value={j}>{j}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
          <Button variant="outline" onClick={loadPapers} disabled={loading}>
            {loading ? 'Loading...' : 'Refresh'}
          </Button>
        </div>

        <div className="text-sm text-muted-foreground">
          Showing {papers.length} of {allPapers.length} papers
        </div>

        <div className="space-y-2 max-h-[600px] overflow-y-auto">
          {papers.map(paper => (
            <div
              key={paper.entry_id}
              className="flex items-start gap-3 rounded-lg border p-3 transition-colors hover:bg-muted/50"
            >
              <button
                onClick={() => toggleStar(paper.entry_id)}
                className="mt-0.5 text-muted-foreground hover:text-accent transition-colors"
                title="Toggle bookmark"
              >
                <Bookmark 
                  className="h-5 w-5 md:h-6 md:w-6" 
                  fill={(paper.tags || []).includes('starred') ? "currentColor" : "none"} 
                  strokeWidth={(paper.tags || []).includes('starred') ? 0 : 1.5}
                  color={(paper.tags || []).includes('starred') ? "#0f766e" : "currentColor"}
                />
              </button>
              <div className="flex-1 min-w-0">
                <div className="font-semibold text-base md:text-[1.15rem] leading-snug">
                  {paper.link ? (
                    <a href={paper.link} target="_blank" rel="noopener noreferrer" className="hover:underline">
                      {paper.title}
                    </a>
                  ) : paper.title}
                </div>
                <div className="mt-1.5 flex flex-wrap gap-2 text-sm text-muted-foreground">
                  {paper.source && <Badge variant="outline" className="text-sm">{paper.source}</Badge>}
                  {paper.published && <span>{paper.published}</span>}
                  {paper.doi && <span className="truncate max-w-[200px]">DOI: {paper.doi}</span>}
                </div>
                {paper.authors && (
                  <div className="mt-1 text-[0.94rem] text-slate-600 line-clamp-1">
                    {paper.authors.length ? paper.authors.join(', ') : 'Unknown authors'}
                  </div>
                )}
                
                {/* Action Buttons */}
                <div className="mt-3 flex flex-wrap gap-2">
                  <Button 
                    variant="outline" 
                    size="sm" 
                    className="h-7 text-xs rounded-full bg-white/70 shadow-sm"
                    onClick={() => setExpanded(prev => ({ ...prev, [paper.entry_id]: !prev[paper.entry_id] }))}
                  >
                    <FileText className="w-3 h-3 mr-1.5" />
                    Abstract
                  </Button>
                  
                  {paper.link && (
                    <a href={paper.link} target="_blank" rel="noopener noreferrer" className="inline-flex items-center justify-center whitespace-nowrap font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 border border-input bg-white/70 shadow-sm hover:bg-accent hover:text-accent-foreground h-7 text-xs rounded-full px-3">
                      <ExternalLink className="w-3 h-3 mr-1.5" /> Source
                    </a>
                  )}
                  
                  {paper.pdf_url && (
                    <a href={paper.pdf_url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center justify-center whitespace-nowrap font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 border border-input bg-white/70 shadow-sm hover:bg-accent hover:text-accent-foreground h-7 text-xs rounded-full px-3">
                      <FileArchive className="w-3 h-3 mr-1.5" /> PDF
                    </a>
                  )}
                  
                  <Button 
                    variant="outline" 
                    size="sm" 
                    className="h-7 text-xs rounded-full bg-orange-50/50 text-orange-800 border-orange-200 hover:bg-orange-100 shadow-sm"
                    onClick={() => setAnalyzeEntry(paper.entry_id)}
                  >
                    <Search className="w-3 h-3 mr-1.5" />
                    Analyze
                  </Button>
                </div>

                {/* Abstract Preview */}
                {expanded[paper.entry_id] && (
                  <div className="mt-3 p-3.5 rounded-2xl border border-border/60 bg-white/60 text-[0.93rem] leading-[1.72] text-slate-700 shadow-inner">
                    {paper.abstract || <span className="italic text-muted-foreground">No abstract available.</span>}
                  </div>
                )}
              </div>
            </div>
          ))}
          {papers.length === 0 && !loading && (
            <div className="text-center text-muted-foreground py-8">No papers found.</div>
          )}
        </div>
      </CardContent>

      <AnalysisDialog 
        open={!!analyzeEntry} 
        onOpenChange={(op) => !op && setAnalyzeEntry(null)}
        entryId={analyzeEntry || undefined}
      />
    </Card>
  );
}
