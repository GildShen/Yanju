import { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { api } from '@/lib/api';

export interface AnalysisDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  entryId?: string;
}

export function AnalysisDialog({ open, onOpenChange, entryId }: AnalysisDialogProps) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState('');
  const [prompt, setPrompt] = useState('Extract methodology, data sources, and key conclusions from this paper.');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleAnalyze = async (formData?: FormData) => {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      let resp;
      if (formData) {
        // Upload PDF route
        formData.append('prompt', prompt);
        resp = await fetch('/api/actions/analyze-pdf', {
          method: 'POST',
          body: formData,
        });
        if (!resp.ok) throw new Error(await resp.text());
        resp = await resp.json();
      } else {
        // Existing entry_id
        resp = await api('/api/actions/analyze', {
          method: 'POST',
          body: JSON.stringify({ entry_id: entryId, prompt }),
        });
      }
      setResult(resp);
    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const fd = new FormData();
      fd.append('file', e.target.files[0]);
      handleAnalyze(fd);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Paper Analysis</DialogTitle>
          <DialogDescription>
            {entryId 
              ? `Analyzing entry ${entryId}` 
              : "Upload a PDF to deep-analyze its contents directly."}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Prompt / Instructions</label>
            <Textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="What do you want to extract?"
              className="resize-none"
            />
          </div>

          <div className="flex gap-2">
            {entryId ? (
              <Button onClick={() => handleAnalyze()} disabled={loading}>
                {loading ? 'Analyzing...' : 'Run Analysis'}
              </Button>
            ) : (
              <div>
                <input 
                  type="file" 
                  accept="application/pdf"
                  className="hidden" 
                  ref={fileInputRef}
                  onChange={handleFileChange} 
                />
                <Button onClick={() => fileInputRef.current?.click()} disabled={loading}>
                  {loading ? 'Uploading & Analyzing...' : 'Upload PDF & Analyze'}
                </Button>
              </div>
            )}
            {entryId && (
              <Button variant="secondary" onClick={() => handleAnalyze()} disabled={loading}>
                Run Methodology Extraction
              </Button>
            )}
          </div>

          {error && <div className="text-red-600 text-sm mt-2">{error}</div>}

          {result && (
            <div className="mt-6 space-y-4">
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline">Model: {result.model || result.answer_model}</Badge>
                {result.cached && <Badge variant="secondary">Cached Result</Badge>}
              </div>
              <div className="rounded-xl border bg-muted/30 p-4 text-[1.05rem] leading-[1.8] whitespace-pre-wrap">
                {result.analysis || result.answer}
              </div>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
