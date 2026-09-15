/**
 * The portfolio content.
 *
 * Adding a project means adding one object to this array. Nothing else in the
 * site needs to change: the tech filter, the counts and the cards all derive
 * from here.
 *
 * `chart` is optional. When present the card renders an interactive figure from
 * real measurements, which is the whole point: a portfolio claim with a number
 * behind it reads differently from one without.
 */

export const profile = {
  name: 'Orlando Jesús Calvo Vargas',
  role: 'AI Engineer',
  tagline: 'LLM systems that are measured, not demoed.',
  blurb:
    'Ten years of production software engineering, currently Full Stack AI Engineer on a direct-to-consumer commerce platform running across 50+ countries. I build RAG pipelines and agentic workflows, and I treat retrieval quality, answer grounding and hallucination rate as numbers I have to defend rather than adjectives in a README.',
  location: 'Costa Rica, remote',
  timezone: 'UTC-6, full US workday overlap',
  english: 'C1 certified',
  links: {
    github: 'https://github.com/chusito27',
    linkedin: 'https://www.linkedin.com/in/orlando-calvo-vargas-207570015/',
    email: 'mailto:orla_ndo21@hotmail.com',
  },
}

export const projects = [
  {
    slug: 'dotnet-docs-rag',
    title: 'Hybrid RAG over technical documentation',
    year: '2026',
    status: 'Live',
    tagline:
      'A retrieval-augmented generation service whose evaluation harness is a first-class component, not an afterthought.',
    summary:
      'Answers questions about the .NET documentation using only passages it can cite, and refuses when the corpus cannot answer. Hybrid retrieval over pgvector and Postgres full-text, fused with weighted Reciprocal Rank Fusion and re-ranked by a cross-encoder. Runs entirely on local models: no API keys, no per-query cost, nothing leaving the machine.',
    metrics: [
      { label: 'Retrieval recall@5', value: '100%', note: 'correct document reached the model every time' },
      { label: 'Citation grounding', value: '100%', note: 'every answer cited passages actually supplied' },
      { label: 'Refusal rate', value: '100%', note: 'on 4 questions the corpus cannot answer' },
      { label: 'Corpus', value: '2,657', note: 'chunks across 310 documents' },
    ],
    highlights: [
      {
        title: 'The ablation disproved part of my own design, and I kept the finding',
        body: 'Textbook unweighted Reciprocal Rank Fusion regressed precision from 100% to 89% recall@1. The lexical arm reaches only 28% alone, and its weak candidates were displacing the correct passage from first place. I swept the fusion weight, documented that hybrid search does not earn its complexity on this corpus, and kept the lexical arm at low weight for exact-token queries the golden set does not stress.',
      },
      {
        title: 'Chose the re-ranker on measured latency, not preference',
        body: 'A Hugging Face cross-encoder matched LLM re-ranking at 100% recall@1 with median latency of 89 ms against 1,460 ms, a 16x reduction, and unlike an LLM it cannot return unparseable output. Both strategies ship behind one interface.',
      },
      {
        title: 'Citations are enforced and validated, not requested',
        body: 'The model may answer only from the supplied passages and must cite a passage id per claim. Cited ids are checked against what was actually sent, so a fabricated citation is caught mechanically rather than trusted.',
      },
      {
        title: 'Provider-agnostic by construction',
        body: 'Local models and an AWS Bedrock path (Titan embeddings, Converse API) sit behind the same two-method interface, so changing provider touches one file. The README states which path produced the measurements and which has not been run against a live account.',
      },
    ],
    tech: [
      'Python',
      'FastAPI',
      'PostgreSQL',
      'pgvector',
      'RAG',
      'Embeddings',
      'Hugging Face',
      'Docker',
      'LLM evaluation',
    ],
    links: {
      repo: 'https://github.com/chusito27/PortfolioOrlandoJesusAI/tree/main/projects/dotnet-docs-rag',
    },
    chart: {
      views: [
        {
          id: 'recall',
          label: 'Retrieval recall@1',
          unit: '%',
          max: 100,
          caption:
            'Same 18 questions, retrieval only. Vector search alone already saturates on this corpus; naive fusion costs precision; the cross-encoder recovers it.',
          bars: [
            { label: 'Vector only', value: 100 },
            { label: 'Keyword only', value: 28 },
            { label: 'Hybrid, weighted RRF', value: 94 },
            { label: 'Hybrid + cross-encoder', value: 100, shipped: true },
          ],
        },
        {
          id: 'latency',
          label: 'Re-ranking latency (p50)',
          unit: ' ms',
          max: 1460,
          caption:
            'Re-ranking step only, measured on the same candidate set. Identical recall, one sixteenth of the latency, and one fewer failure mode.',
          bars: [
            { label: 'LLM re-ranking', value: 1460 },
            { label: 'Cross-encoder', value: 89, shipped: true },
          ],
        },
      ],
    },
  },
]
