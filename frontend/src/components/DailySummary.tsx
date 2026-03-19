import { useState, useEffect, useCallback } from 'react';
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select';
import { streamApi, type SummaryResult } from '@/lib/api';

export function DailySummary({ isInline = false }: { isInline?: boolean }) {
  const [date, setDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [language, setLanguage] = useState('zh-TW');
  const [starredOnly, setStarredOnly] = useState(false);
  const [summary, setSummary] = useState('');
  const [meta, setMeta] = useState<Partial<SummaryResult>>({});
  const [loading, setLoading] = useState(false);
  const [topicInput, setTopicInput] = useState('');

  // Load research topic on mount
  useEffect(() => {
    fetch('/api/settings/research-topic')
      .then(r => r.ok ? r.json() : { topic: '' })
      .then(d => { setTopicInput(d.topic || ''); })
      .catch(() => {});
  }, []);

  const loadSummary = useCallback(async (forceRefresh = false) => {
    setLoading(true);
    setSummary('');
    setMeta({});
    let collected = '';
    try {
      await streamApi('/api/actions/today-summary/stream', {
        language,
        model: 'gpt-5-mini',
        limit: 15,
        target_date: date,
        force_refresh: forceRefresh,
        starred_only: starredOnly,
      }, {
        onMeta(data) {
          setMeta(data as Partial<SummaryResult>);
        },
        onDelta(data) {
          collected += data.text || '';
          setSummary(collected);
        },
        onDone(data) {
          setMeta(prev => ({ ...prev, ...data, cached: (data as Record<string, unknown>).cached as boolean }));
        },
        onError(data) {
          setSummary(`Error: ${data.detail}`);
        },
      });
    } catch (err) {
      setSummary(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(false);
    }
  }, [date, language, starredOnly]);

  useEffect(() => { loadSummary(); }, [loadSummary]);

  const saveTopic = async () => {
    try {
      const res = await fetch('/api/settings/research-topic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: topicInput }),
      });
      if (res.ok) {
        const data = await res.json();
        setTopicInput(data.topic || topicInput);
      }
    } catch {}
  };

  const Content = (
    <div className="space-y-4">
        {/* Controls row */}
        <div className="flex flex-wrap items-center gap-3">
          <Input
            type="date"
            value={date}
            onChange={e => setDate(e.target.value)}
            className="w-44"
          />
          <Select value={language} onValueChange={v => setLanguage(v || 'zh-TW')}>
            <SelectTrigger className="w-48">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="zh-TW">Traditional Chinese</SelectItem>
              <SelectItem value="en">English</SelectItem>
            </SelectContent>
          </Select>
          <Button onClick={() => loadSummary(false)} disabled={loading}>
            {loading ? 'Loading...' : 'Load Summary'}
          </Button>
          <Button variant="outline" onClick={() => loadSummary(true)} disabled={loading}>
            Regenerate
          </Button>
        </div>

        {/* Starred toggle */}
        <div className="flex items-center gap-2">
          <Switch
            id="starred-only"
            checked={starredOnly}
            onCheckedChange={(checked) => { setStarredOnly(checked); }}
          />
          <Label htmlFor="starred-only" className="text-sm text-muted-foreground">
            Summarize starred papers only (-B)
          </Label>
        </div>

        {/* Topic row */}
        <div className="flex items-center gap-2">
          <Input
            placeholder="Set your research topic, e.g. human-AI collaboration in knowledge management"
            value={topicInput}
            onChange={e => setTopicInput(e.target.value)}
            className="flex-1"
          />
          <Button variant="outline" size="sm" onClick={saveTopic}>
            Save Topic
          </Button>
        </div>

        {/* Summary output */}
        <div className="min-h-[220px] rounded-[20px] border border-[rgba(180,83,9,0.16)] bg-white/60 p-5 md:p-6 text-[1.05rem] leading-[1.8] whitespace-pre-wrap shadow-inner backdrop-blur-sm">
          {summary || (loading ? 'Loading summary...' : 'No summary loaded.')}
        </div>

        {/* Meta info */}
        {meta.date && (
          <div className="flex flex-wrap gap-2 text-sm">
            <Badge variant="secondary">{meta.date}</Badge>
            <Badge variant="secondary">{meta.language}</Badge>
            <Badge variant="secondary">{meta.paper_count ?? 0} papers</Badge>
            <Badge variant="secondary">{meta.selection_mode}</Badge>
            {meta.cached !== undefined && (
              <Badge variant={meta.cached ? 'outline' : 'default'}>
                {meta.cached ? 'cached' : 'generated'}
              </Badge>
            )}
            {meta.topic && <Badge variant="outline">Topic: {meta.topic}</Badge>}
          </div>
        )}
    </div>
  );

  if (isInline) {
    return Content;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xl">Daily Summary</CardTitle>
        <CardDescription>
          Select a date, language, and research topic to generate a compact research-note style summary.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {Content}
      </CardContent>
    </Card>
  );
}
