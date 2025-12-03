"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  Search,
  Loader2,
  Clock,
  Database,
  Globe,
  Zap,
  FileText,
  ChevronDown,
  ChevronUp,
  Trash2,
  ExternalLink,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface Source {
  content: string;
  source: string;
  score: number;
  metadata?: Record<string, unknown>;
}

interface RoutingDecision {
  intent?: string;
  confidence?: number;
  reason?: string;
  requires_web?: boolean;
  search_query?: string;
}

interface Timing {
  total_ms?: number;
  cache_lookup_ms?: number;
  routing_ms?: number;
  retrieval_ms?: number;
  generation_ms?: number;
}

interface CacheMetadata {
  similarity_score?: number;
  cached_question?: string;
}

interface SearchResult {
  query: string;
  response: string;
  sources: Source[];
  cache_hit: boolean;
  cache_metadata: CacheMetadata;
  routing_decision: RoutingDecision;
  timing: Timing;
}

interface CacheStats {
  total_queries: number;
  cache_hits: number;
  cache_misses: number;
  hit_rate_percent: number;
  total_time_saved_ms: number;
  avg_time_saved_ms: number;
  cache_size: number;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const EXAMPLE_QUERIES = [
  "What are the main risk factors mentioned in the 10-K filings?",
  "How do I build an AI agent with OpenAI?",
  "Latest Nvidia earnings news",
];

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<SearchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null);
  const [history, setHistory] = useState<SearchResult[]>([]);
  const [showDebug, setShowDebug] = useState(false);
  const [expandedSources, setExpandedSources] = useState<number[]>([]);
  const [cacheCleared, setCacheCleared] = useState(false);

  const handleSearch = async (searchQuery: string) => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    setError(null);
    setQuery(searchQuery);

    try {
      const response = await fetch(`${API_URL}/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: searchQuery.trim(), use_cache: true }),
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      const data: SearchResult = await response.json();
      setResult(data);
      setHistory((prev) => [data, ...prev.slice(0, 9)]);
      setExpandedSources([]);
      fetchCacheStats();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSearch(query);
  };

  const fetchCacheStats = async () => {
    try {
      const response = await fetch(`${API_URL}/cache/stats`, {
        cache: "no-store",
      });
      if (response.ok) {
        setCacheStats(await response.json());
      }
    } catch {
      // Silently fail for stats
    }
  };

  const clearCache = async () => {
    try {
      const response = await fetch(`${API_URL}/cache/clear`, {
        method: "POST",
        cache: "no-store",
      });
      if (response.ok) {
        setCacheCleared(true);
        setCacheStats(null);
        setResult(null); // Clear the displayed result
        setHistory([]); // Clear history since cache is cleared
        fetchCacheStats();
        // Hide the success message after 3 seconds
        setTimeout(() => setCacheCleared(false), 3000);
      }
    } catch {
      // Silently fail
    }
  };

  const toggleSource = (index: number) => {
    setExpandedSources((prev) =>
      prev.includes(index)
        ? prev.filter((i) => i !== index)
        : [...prev, index]
    );
  };

  const getSourceIcon = (sourceType: string) => {
    if (sourceType === "10k_data" || sourceType === "local_10k") {
      return <FileText className="h-4 w-4" />;
    }
    if (sourceType === "web_search") {
      return <Globe className="h-4 w-4" />;
    }
    return <Database className="h-4 w-4" />;
  };

  const getSourceLabel = (source: Source) => {
    // Check for URL in metadata
    const url = source.metadata?.url as string | undefined;
    const title = source.metadata?.title as string | undefined;
    const fileName = source.metadata?.file_name as string | undefined;

    if (title) return title;
    if (fileName) return fileName;
    if (url) {
      try {
        return new URL(url).hostname;
      } catch {
        return url;
      }
    }
    return source.source.replace(/_/g, " ");
  };

  const getSourceUrl = (source: Source): string | null => {
    const url = source.metadata?.url as string | undefined;
    const link = source.metadata?.link as string | undefined;
    return url || link || null;
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8 max-w-5xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-2">
            Agentic Search Engine
          </h1>
          <p className="text-slate-600 dark:text-slate-400">
            RAG-powered semantic search with intelligent caching and query routing
          </p>
        </div>

        {/* Search Form */}
        <form onSubmit={handleSubmit} className="mb-6">
          <div className="flex gap-2">
            <Input
              type="text"
              placeholder="Ask a question about 10-K filings, AI, or search the web..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="flex-1 h-12 text-lg"
              disabled={loading}
            />
            <Button type="submit" disabled={loading || !query.trim()} size="lg">
              {loading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Search className="h-5 w-5" />
              )}
            </Button>
          </div>
        </form>

        {/* Example Queries and Clear Cache */}
        <div className="mb-8 flex flex-wrap items-start justify-between gap-4">
          <div className="flex-1">
            <p className="text-sm text-slate-500 mb-2">Try these examples:</p>
            <div className="flex flex-wrap gap-2">
              {EXAMPLE_QUERIES.map((example, idx) => (
                <Button
                  key={idx}
                  variant="outline"
                  size="sm"
                  onClick={() => handleSearch(example)}
                  disabled={loading}
                  className="text-xs"
                >
                  {example}
                </Button>
              ))}
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={clearCache}
            className="text-xs text-red-500 hover:text-red-700 hover:bg-red-50"
          >
            <Trash2 className="h-3 w-3 mr-1" />
            Clear Cache
          </Button>
        </div>

        {/* Cache Cleared Success */}
        {cacheCleared && (
          <Card className="mb-6 border-green-200 bg-green-50">
            <CardContent className="pt-6">
              <p className="text-green-600 flex items-center gap-2">
                <Zap className="h-4 w-4" />
                Cache cleared successfully!
              </p>
            </CardContent>
          </Card>
        )}

        {/* Error */}
        {error && (
          <Card className="mb-6 border-red-200 bg-red-50">
            <CardContent className="pt-6">
              <p className="text-red-600">{error}</p>
              <p className="text-sm text-red-500 mt-2">
                Make sure the API server is running: uvicorn api:app --port 8000
              </p>
            </CardContent>
          </Card>
        )}

        {/* Result */}
        {result && (
          <div className="space-y-6">
            {/* Cache Status */}
            {result.cache_hit ? (
              <Card className="border-green-200 bg-green-50 dark:bg-green-900/20">
                <CardContent className="pt-6">
                  <div className="flex items-center gap-2 text-green-700 dark:text-green-400">
                    <Zap className="h-5 w-5" />
                    <span className="font-semibold">Cache Hit!</span>
                    <span className="text-sm">Response retrieved from semantic cache.</span>
                  </div>
                  {result.cache_metadata?.similarity_score && (
                    <p className="text-sm text-green-600 dark:text-green-500 mt-1">
                      Similarity: {(result.cache_metadata.similarity_score * 100).toFixed(1)}%
                      {result.cache_metadata.cached_question && (
                        <span className="ml-2">
                          | Cached: &quot;{result.cache_metadata.cached_question.slice(0, 60)}...&quot;
                        </span>
                      )}
                    </p>
                  )}
                </CardContent>
              </Card>
            ) : (
              <Card className="border-blue-200 bg-blue-50 dark:bg-blue-900/20">
                <CardContent className="pt-6">
                  <div className="flex items-center gap-2 text-blue-700 dark:text-blue-400">
                    <Database className="h-5 w-5" />
                    <span className="font-semibold">Cache Miss</span>
                    <span className="text-sm">Performed full RAG search.</span>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Answer */}
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg">Answer</CardTitle>
                  <Badge variant="outline" className="text-xs">
                    <Clock className="h-3 w-3 mr-1" />
                    {result.timing?.total_ms?.toFixed(0) || 0}ms
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="prose prose-slate dark:prose-invert max-w-none prose-headings:font-semibold prose-h1:text-2xl prose-h2:text-xl prose-h3:text-lg prose-p:text-slate-700 dark:prose-p:text-slate-300 prose-li:text-slate-700 dark:prose-li:text-slate-300 prose-strong:text-slate-900 dark:prose-strong:text-slate-100 prose-code:bg-slate-100 dark:prose-code:bg-slate-800 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-pre:bg-slate-100 dark:prose-pre:bg-slate-800">
                  <ReactMarkdown>{result.response}</ReactMarkdown>
                </div>
              </CardContent>
            </Card>

            {/* Debug Info Toggle */}
            <div className="flex justify-center">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowDebug(!showDebug)}
              >
                {showDebug ? "Hide" : "Show"} Debug Info
                {showDebug ? (
                  <ChevronUp className="ml-1 h-4 w-4" />
                ) : (
                  <ChevronDown className="ml-1 h-4 w-4" />
                )}
              </Button>
            </div>

            {/* Debug Info */}
            {showDebug && (
              <div className="grid md:grid-cols-2 gap-4">
                {/* Routing Decision */}
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Routing Decision</CardTitle>
                  </CardHeader>
                  <CardContent className="text-sm space-y-1">
                    <div>
                      <span className="text-slate-500">Intent:</span>{" "}
                      <code className="bg-slate-100 dark:bg-slate-800 px-1 rounded">
                        {result.routing_decision?.intent || "N/A"}
                      </code>
                    </div>
                    <div>
                      <span className="text-slate-500">Confidence:</span>{" "}
                      {((result.routing_decision?.confidence || 0) * 100).toFixed(0)}%
                    </div>
                    <div>
                      <span className="text-slate-500">Reason:</span>{" "}
                      {result.routing_decision?.reason || "N/A"}
                    </div>
                    <div>
                      <span className="text-slate-500">Web Search:</span>{" "}
                      {result.routing_decision?.requires_web ? "Yes" : "No"}
                    </div>
                  </CardContent>
                </Card>

                {/* Timing */}
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Timing Breakdown</CardTitle>
                  </CardHeader>
                  <CardContent className="text-sm space-y-1">
                    <div>
                      <span className="text-slate-500">Total:</span>{" "}
                      {result.timing?.total_ms?.toFixed(0) || 0}ms
                    </div>
                    <div>
                      <span className="text-slate-500">Cache Lookup:</span>{" "}
                      {result.timing?.cache_lookup_ms?.toFixed(0) || 0}ms
                    </div>
                    <div>
                      <span className="text-slate-500">Routing:</span>{" "}
                      {result.timing?.routing_ms?.toFixed(0) || 0}ms
                    </div>
                    <div>
                      <span className="text-slate-500">Retrieval:</span>{" "}
                      {result.timing?.retrieval_ms?.toFixed(0) || 0}ms
                    </div>
                    <div>
                      <span className="text-slate-500">Generation:</span>{" "}
                      {result.timing?.generation_ms?.toFixed(0) || 0}ms
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Sources */}
            {result.sources && result.sources.length > 0 && (
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">
                    Sources ({result.sources.length} documents)
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {result.sources.map((source, idx) => {
                    const sourceUrl = getSourceUrl(source);
                    const sourceLabel = getSourceLabel(source);

                    return (
                      <div
                        key={idx}
                        className="border rounded-lg overflow-hidden"
                      >
                        <button
                          onClick={() => toggleSource(idx)}
                          className="w-full flex items-center justify-between p-3 hover:bg-slate-50 dark:hover:bg-slate-800 text-left"
                        >
                          <div className="flex items-center gap-2 flex-1 min-w-0">
                            {getSourceIcon(source.source)}
                            <span className="text-sm font-medium truncate">
                              {sourceLabel}
                            </span>
                            {sourceUrl && (
                              <a
                                href={sourceUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                onClick={(e) => e.stopPropagation()}
                                className="text-blue-500 hover:text-blue-700 flex-shrink-0"
                              >
                                <ExternalLink className="h-3 w-3" />
                              </a>
                            )}
                            <Badge variant="outline" className="text-xs flex-shrink-0">
                              {source.score.toFixed(2)}
                            </Badge>
                          </div>
                          {expandedSources.includes(idx) ? (
                            <ChevronUp className="h-4 w-4 flex-shrink-0 ml-2" />
                          ) : (
                            <ChevronDown className="h-4 w-4 flex-shrink-0 ml-2" />
                          )}
                        </button>
                        {expandedSources.includes(idx) && (
                          <div className="p-3 pt-0 border-t bg-slate-50 dark:bg-slate-800/50">
                            <div className="mb-2">
                              <span className="text-xs text-slate-500">Type: </span>
                              <Badge variant="secondary" className="text-xs">
                                {source.source}
                              </Badge>
                              {sourceUrl && (
                                <a
                                  href={sourceUrl}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="ml-2 text-xs text-blue-500 hover:underline inline-flex items-center gap-1"
                                >
                                  Open source <ExternalLink className="h-3 w-3" />
                                </a>
                              )}
                            </div>
                            <p className="text-sm text-slate-600 dark:text-slate-400 whitespace-pre-wrap">
                              {source.content.slice(0, 500)}
                              {source.content.length > 500 && "..."}
                            </p>
                            {source.metadata && Object.keys(source.metadata).length > 0 && (
                              <details className="mt-2">
                                <summary className="text-xs text-slate-500 cursor-pointer hover:text-slate-700">
                                  View metadata
                                </summary>
                                <pre className="mt-1 text-xs bg-slate-100 dark:bg-slate-900 p-2 rounded overflow-x-auto">
                                  {JSON.stringify(source.metadata, null, 2)}
                                </pre>
                              </details>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </CardContent>
              </Card>
            )}
          </div>
        )}

        {/* Cache Stats */}
        {cacheStats && (
          <Card className="mt-6">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-medium">Cache Statistics</CardTitle>
                <Button variant="ghost" size="sm" onClick={clearCache}>
                  <Trash2 className="h-4 w-4 mr-1" />
                  Clear
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-4 gap-4 text-center">
                <div>
                  <div className="text-2xl font-bold">
                    {cacheStats.total_queries || 0}
                  </div>
                  <div className="text-xs text-slate-500">Total</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-green-600">
                    {cacheStats.cache_hits || 0}
                  </div>
                  <div className="text-xs text-slate-500">Hits</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-slate-600">
                    {cacheStats.cache_misses || 0}
                  </div>
                  <div className="text-xs text-slate-500">Misses</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-blue-600">
                    {(cacheStats.hit_rate_percent || 0).toFixed(1)}%
                  </div>
                  <div className="text-xs text-slate-500">Hit Rate</div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* History */}
        {history.length > 1 && (
          <Card className="mt-6">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Recent Searches</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {history.slice(1).map((item, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between p-2 rounded bg-slate-50 dark:bg-slate-800 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700"
                    onClick={() => handleSearch(item.query)}
                  >
                    <span className="text-sm truncate flex-1">{item.query}</span>
                    <div className="flex items-center gap-2 ml-2">
                      <Badge variant={item.cache_hit ? "success" : "outline"} className="text-xs">
                        {item.cache_hit ? "cached" : "searched"}
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        {item.timing?.total_ms?.toFixed(0) || 0}ms
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Footer */}
        <div className="text-center mt-8 text-sm text-slate-500">
          <p>Powered by Nomic Embeddings, Qdrant Vector DB, and OpenAI</p>
        </div>
      </div>
    </main>
  );
}
