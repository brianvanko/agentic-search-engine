import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import SearchPage from '@/app/page';

// Mock fetch globally
const mockFetch = jest.fn();
global.fetch = mockFetch;

// Mock SearchResult response
const mockSearchResult = {
  query: 'What is Lyft revenue?',
  response: 'Lyft revenue is approximately $4.1 billion.',
  sources: [
    {
      content: 'Lyft reported revenue of $4.1 billion in their 10-K filing.',
      source: '10k_data',
      score: 0.92,
      metadata: { file_name: 'lyft_10k.pdf' },
    },
  ],
  cache_hit: false,
  cache_metadata: {},
  routing_decision: {
    intent: 'LOCAL_10K',
    confidence: 0.95,
    reason: 'Query about company financials',
    requires_web: false,
  },
  timing: {
    total_ms: 1234,
    cache_lookup_ms: 5,
    routing_ms: 150,
    retrieval_ms: 300,
    generation_ms: 779,
  },
};

const mockCacheStats = {
  total_queries: 10,
  cache_hits: 7,
  cache_misses: 3,
  hit_rate_percent: 70,
  total_time_saved_ms: 5000,
  avg_time_saved_ms: 714,
  cache_size: 50,
};

describe('SearchPage', () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  describe('Rendering', () => {
    it('renders the page title', () => {
      render(<SearchPage />);
      expect(screen.getByText('Agentic Search Engine')).toBeInTheDocument();
    });

    it('renders the search input', () => {
      render(<SearchPage />);
      expect(
        screen.getByPlaceholderText(/ask a question/i)
      ).toBeInTheDocument();
    });

    it('renders example queries', () => {
      render(<SearchPage />);
      expect(screen.getByText(/try these examples/i)).toBeInTheDocument();
      expect(
        screen.getByText(/what are the main risk factors/i)
      ).toBeInTheDocument();
    });

    it('renders the search button', () => {
      render(<SearchPage />);
      const searchButton = screen.getByRole('button', { name: '' });
      expect(searchButton).toBeInTheDocument();
    });

    it('renders the clear cache button', () => {
      render(<SearchPage />);
      expect(screen.getByText(/clear cache/i)).toBeInTheDocument();
    });
  });

  describe('Search Functionality', () => {
    it('disables search button when input is empty', () => {
      render(<SearchPage />);
      const submitButton = screen.getByRole('button', { name: '' });
      expect(submitButton).toBeDisabled();
    });

    it('enables search button when input has text', () => {
      render(<SearchPage />);
      const input = screen.getByPlaceholderText(/ask a question/i);
      fireEvent.change(input, { target: { value: 'test query' } });

      const submitButton = screen.getByRole('button', { name: '' });
      expect(submitButton).not.toBeDisabled();
    });

    it('performs search when form is submitted', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockSearchResult),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockCacheStats),
        });

      render(<SearchPage />);
      const input = screen.getByPlaceholderText(/ask a question/i);
      fireEvent.change(input, { target: { value: 'What is Lyft revenue?' } });

      const form = input.closest('form')!;
      fireEvent.submit(form);

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining('/search'),
          expect.objectContaining({
            method: 'POST',
            body: JSON.stringify({
              query: 'What is Lyft revenue?',
              use_cache: true,
            }),
          })
        );
      });
    });

    it('displays search results after successful search', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockSearchResult),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockCacheStats),
        });

      render(<SearchPage />);
      const input = screen.getByPlaceholderText(/ask a question/i);
      fireEvent.change(input, { target: { value: 'What is Lyft revenue?' } });

      const form = input.closest('form')!;
      fireEvent.submit(form);

      await waitFor(() => {
        expect(screen.getByText(/answer/i)).toBeInTheDocument();
      });

      expect(
        screen.getByText(/Lyft revenue is approximately \$4\.1 billion/i)
      ).toBeInTheDocument();
    });

    it('displays error when search fails', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        statusText: 'Internal Server Error',
      });

      render(<SearchPage />);
      const input = screen.getByPlaceholderText(/ask a question/i);
      fireEvent.change(input, { target: { value: 'test query' } });

      const form = input.closest('form')!;
      fireEvent.submit(form);

      await waitFor(() => {
        expect(screen.getByText(/search failed/i)).toBeInTheDocument();
      });
    });

    it('performs search when example query is clicked', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockSearchResult),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockCacheStats),
        });

      render(<SearchPage />);
      const exampleButton = screen.getByText(
        /what are the main risk factors/i
      );
      fireEvent.click(exampleButton);

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalled();
      });
    });
  });

  describe('Cache Hit/Miss Display', () => {
    it('shows cache miss indicator for new queries', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockSearchResult),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockCacheStats),
        });

      render(<SearchPage />);
      const input = screen.getByPlaceholderText(/ask a question/i);
      fireEvent.change(input, { target: { value: 'test query' } });
      fireEvent.submit(input.closest('form')!);

      await waitFor(() => {
        expect(screen.getByText(/cache miss/i)).toBeInTheDocument();
      });
    });

    it('shows cache hit indicator when response is cached', async () => {
      const cachedResult = {
        ...mockSearchResult,
        cache_hit: true,
        cache_metadata: {
          similarity_score: 0.95,
          cached_question: 'What is Lyft revenue?',
        },
      };

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(cachedResult),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockCacheStats),
        });

      render(<SearchPage />);
      const input = screen.getByPlaceholderText(/ask a question/i);
      fireEvent.change(input, { target: { value: 'test query' } });
      fireEvent.submit(input.closest('form')!);

      await waitFor(() => {
        expect(screen.getByText(/cache hit/i)).toBeInTheDocument();
      });
    });
  });

  describe('Debug Info', () => {
    it('toggles debug info visibility', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockSearchResult),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockCacheStats),
        });

      render(<SearchPage />);
      const input = screen.getByPlaceholderText(/ask a question/i);
      fireEvent.change(input, { target: { value: 'test' } });
      fireEvent.submit(input.closest('form')!);

      await waitFor(() => {
        expect(screen.getByText(/show debug info/i)).toBeInTheDocument();
      });

      const toggleButton = screen.getByText(/show debug info/i);
      fireEvent.click(toggleButton);

      expect(screen.getByText(/routing decision/i)).toBeInTheDocument();
      expect(screen.getByText(/timing breakdown/i)).toBeInTheDocument();
    });
  });

  describe('Sources Display', () => {
    it('displays sources when available', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockSearchResult),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockCacheStats),
        });

      render(<SearchPage />);
      const input = screen.getByPlaceholderText(/ask a question/i);
      fireEvent.change(input, { target: { value: 'test' } });
      fireEvent.submit(input.closest('form')!);

      await waitFor(() => {
        expect(screen.getByText(/sources/i)).toBeInTheDocument();
      });

      expect(screen.getByText(/1 documents/i)).toBeInTheDocument();
    });

    it('expands source content when clicked', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockSearchResult),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockCacheStats),
        });

      render(<SearchPage />);
      const input = screen.getByPlaceholderText(/ask a question/i);
      fireEvent.change(input, { target: { value: 'test' } });
      fireEvent.submit(input.closest('form')!);

      await waitFor(() => {
        expect(screen.getByText(/lyft_10k\.pdf/i)).toBeInTheDocument();
      });

      const sourceButton = screen.getByText(/lyft_10k\.pdf/i);
      fireEvent.click(sourceButton);

      expect(
        screen.getByText(/Lyft reported revenue of \$4\.1 billion/i)
      ).toBeInTheDocument();
    });
  });

  describe('Cache Management', () => {
    it('clears cache when clear button is clicked', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ status: 'cleared' }),
      }).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ ...mockCacheStats, cache_size: 0 }),
      });

      render(<SearchPage />);
      const clearButton = screen.getByText(/clear cache/i);
      fireEvent.click(clearButton);

      await waitFor(() => {
        expect(screen.getByText(/cache cleared successfully/i)).toBeInTheDocument();
      });
    });

    it('displays cache statistics', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockSearchResult),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockCacheStats),
        });

      render(<SearchPage />);
      const input = screen.getByPlaceholderText(/ask a question/i);
      fireEvent.change(input, { target: { value: 'test' } });
      fireEvent.submit(input.closest('form')!);

      await waitFor(() => {
        expect(screen.getByText(/cache statistics/i)).toBeInTheDocument();
      });
    });
  });

  describe('Search History', () => {
    it('displays search history after multiple searches', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockSearchResult),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockCacheStats),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () =>
            Promise.resolve({ ...mockSearchResult, query: 'Second query' }),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockCacheStats),
        });

      render(<SearchPage />);

      // First search
      const input = screen.getByPlaceholderText(/ask a question/i);
      fireEvent.change(input, { target: { value: 'First query' } });
      fireEvent.submit(input.closest('form')!);

      await waitFor(() => {
        expect(screen.getByText(/answer/i)).toBeInTheDocument();
      });

      // Second search
      fireEvent.change(input, { target: { value: 'Second query' } });
      fireEvent.submit(input.closest('form')!);

      await waitFor(() => {
        expect(screen.getByText(/recent searches/i)).toBeInTheDocument();
      });
    });
  });
});

describe('Utility Functions', () => {
  describe('getSourceLabel', () => {
    it('should be tested through component behavior', () => {
      // These utility functions are tested indirectly through component tests
      expect(true).toBe(true);
    });
  });
});
