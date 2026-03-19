import { useState } from 'react';
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Accordion, AccordionContent, AccordionItem, AccordionTrigger,
} from '@/components/ui/accordion';
import { api } from '@/lib/api';

export function Workbench() {
  const [output, setOutput] = useState('Ready.');
  const [loading, setLoading] = useState(false);

  // Catalog fields
  const [pdfDir, setPdfDir] = useState('papers/tmp');
  const [embedModel, setEmbedModel] = useState('text-embedding-3-small');
  const [dimensions, setDimensions] = useState('');

  // Import fields
  const [doiFile, setDoiFile] = useState('dois.txt');
  const [importUrl, setImportUrl] = useState('');

  // Search fields
  const [searchQuery, setSearchQuery] = useState('');
  const [searchLimit, setSearchLimit] = useState('5');

  // Ask fields
  const [askQuery, setAskQuery] = useState('What themes appear in my current embedded papers?');
  const [askModel, setAskModel] = useState('gpt-5-mini');
  const [askTopK, setAskTopK] = useState('5');

  const run = async (path: string, body: Record<string, unknown>) => {
    setLoading(true);
    setOutput('Running...');
    try {
      const data = await api(path, { method: 'POST', body: JSON.stringify(body) });
      setOutput(JSON.stringify(data, null, 2));
    } catch (err) {
      setOutput(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xl">Workbench</CardTitle>
        <CardDescription>
          Group ingestion, cataloging, import, search, and Q&A into one collapsible workbench.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* @ts-ignore Component prop typing issue from shadcn/radix */}
        <Accordion type="multiple" defaultValue={['ingest']}>
          {/* Ingest & Digest */}
          <AccordionItem value="ingest">
            <AccordionTrigger>Ingest & Digest</AccordionTrigger>
            <AccordionContent className="space-y-3 pt-2">
              <div className="flex gap-2">
                <Button onClick={() => run('/api/actions/ingest', {})} disabled={loading}>
                  Run Ingest
                </Button>
                <Button variant="secondary" onClick={() => run('/api/actions/digest', { days: 7 })} disabled={loading}>
                  Run Digest
                </Button>
              </div>
            </AccordionContent>
          </AccordionItem>

          {/* Catalog PDFs */}
          <AccordionItem value="catalog">
            <AccordionTrigger>Catalog PDFs</AccordionTrigger>
            <AccordionContent className="space-y-3 pt-2">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-muted-foreground">PDF directory</label>
                  <Input value={pdfDir} onChange={e => setPdfDir(e.target.value)} />
                </div>
                <div>
                  <label className="text-xs text-muted-foreground">Embedding model</label>
                  <Input value={embedModel} onChange={e => setEmbedModel(e.target.value)} />
                </div>
              </div>
              <div className="mt-3">
                <label className="text-xs text-muted-foreground">Dimensions</label>
                <Input value={dimensions} onChange={e => setDimensions(e.target.value)} type="number" placeholder="e.g. 1536" />
              </div>
              <div className="flex gap-2">
                <Button onClick={() => run('/api/actions/catalog', { pdf_dir: pdfDir, embed: false })} disabled={loading}>
                  Catalog PDFs
                </Button>
                <Button variant="secondary" onClick={() => run('/api/actions/catalog', { pdf_dir: pdfDir, embed: true, embedding_model: embedModel, dimensions: dimensions ? parseInt(dimensions) : null })} disabled={loading}>
                  Catalog + Embed
                </Button>
              </div>
            </AccordionContent>
          </AccordionItem>

          {/* Import */}
          <AccordionItem value="import">
            <AccordionTrigger>Import DOIs / URL</AccordionTrigger>
            <AccordionContent className="space-y-3 pt-2">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-muted-foreground">DOI file</label>
                  <Input value={doiFile} onChange={e => setDoiFile(e.target.value)} />
                </div>
                <div>
                  <label className="text-xs text-muted-foreground">Import URL</label>
                  <Input value={importUrl} onChange={e => setImportUrl(e.target.value)} placeholder="https://arxiv.org/abs/..." />
                </div>
              </div>
              <div className="flex gap-2">
                <Button onClick={() => run('/api/actions/import-dois', { doi_file: doiFile })} disabled={loading}>
                  Import DOIs
                </Button>
                <Button variant="secondary" onClick={() => run('/api/actions/import-url', { url: importUrl })} disabled={loading || !importUrl}>
                  Import URL
                </Button>
              </div>
            </AccordionContent>
          </AccordionItem>

          {/* Semantic Search */}
          <AccordionItem value="search">
            <AccordionTrigger>Semantic Search</AccordionTrigger>
            <AccordionContent className="space-y-3 pt-2">
              <Input placeholder="Search query..." value={searchQuery} onChange={e => setSearchQuery(e.target.value)} />
              <div>
                <label className="text-xs text-muted-foreground">Limit</label>
                <Input value={searchLimit} onChange={e => setSearchLimit(e.target.value)} className="w-20" />
              </div>
              <Button onClick={() => run('/api/actions/search', { query: searchQuery, limit: parseInt(searchLimit) })} disabled={loading || !searchQuery}>
                Semantic Search
              </Button>
            </AccordionContent>
          </AccordionItem>

          {/* Ask */}
          <AccordionItem value="ask">
            <AccordionTrigger>Ask</AccordionTrigger>
            <AccordionContent className="space-y-3 pt-2">
              <textarea
                className="w-full rounded-md border bg-background px-3 py-2 text-sm min-h-[80px]"
                value={askQuery}
                onChange={e => setAskQuery(e.target.value)}
              />
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-muted-foreground">Answer model</label>
                  <Input value={askModel} onChange={e => setAskModel(e.target.value)} />
                </div>
                <div>
                  <label className="text-xs text-muted-foreground">Top K</label>
                  <Input value={askTopK} onChange={e => setAskTopK(e.target.value)} className="w-20" />
                </div>
              </div>
              <Button variant="secondary" onClick={() => run('/api/actions/ask', { query: askQuery, answer_model: askModel, top_k: parseInt(askTopK) })} disabled={loading || !askQuery}>
                Ask
              </Button>
            </AccordionContent>
          </AccordionItem>
        </Accordion>

        {/* Output */}
        <div className="rounded-lg border bg-muted/30 p-4 text-xs font-mono whitespace-pre-wrap max-h-[300px] overflow-y-auto">
          {output}
        </div>
      </CardContent>
    </Card>
  );
}
