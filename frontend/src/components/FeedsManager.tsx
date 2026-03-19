import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { api } from '@/lib/api';

export function FeedsManager() {
  const [feeds, setFeeds] = useState('');
  const [isEditing, setIsEditing] = useState(false);
  const [saving, setSaving] = useState(false);

  const loadFeeds = async () => {
    try {
      const resp = await api<{ text: string }>('/api/actions/manage-list?name=feeds.txt');
      setFeeds(resp.text || '');
    } catch {}
  };

  useEffect(() => {
    loadFeeds();
  }, []);

  const saveFeeds = async () => {
    setSaving(true);
    try {
      await api('/api/actions/manage-list?name=feeds.txt', {
        method: 'POST',
        body: JSON.stringify({ text: feeds }),
      });
      setIsEditing(false);
    } catch (err) {
      console.error(err);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card className="border border-border/50 bg-card/60 shadow-sm backdrop-blur-sm">
      <CardHeader className="py-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">RSS Feeds</CardTitle>
          {isEditing ? (
            <div className="space-x-2">
              <Button size="sm" variant="outline" onClick={() => { setIsEditing(false); loadFeeds(); }}>Cancel</Button>
              <Button size="sm" onClick={saveFeeds} disabled={saving}>Save</Button>
            </div>
          ) : (
            <Button size="sm" variant="ghost" className="text-muted-foreground" onClick={() => setIsEditing(true)}>Edit</Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="pb-4">
        {isEditing ? (
          <Textarea 
            value={feeds}
            onChange={(e) => setFeeds(e.target.value)}
            className="min-h-[150px] font-mono text-sm"
          />
        ) : (
          <div 
            className="min-h-[100px] whitespace-pre-wrap rounded-md border border-dashed p-3 text-sm text-muted-foreground hover:bg-muted/30 cursor-pointer"
            onDoubleClick={() => setIsEditing(true)}
            title="Double click to edit"
          >
            {feeds || "No feeds configured. Double click to add some."}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
