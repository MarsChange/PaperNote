export interface PaperSource {
  id: string
  block_id?: string
  chunk_id?: string
  parent_chunk_id?: string
  root_chunk_id?: string
  chunk_level?: number
  type: string
  page_number: number
  title: string
  section?: string
  content: string
  score?: number
  rerank_score?: number
  rrf_rank?: number
  asset_url?: string
  merged_from_children?: boolean
  merged_child_count?: number
}

export interface RagStep {
  icon: string
  label: string
  detail?: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  sources?: PaperSource[]
  ragSteps?: RagStep[]
  ragTrace?: Record<string, unknown>
}

export interface PaperFile {
  id: string
  name: string
  filename?: string
  status: 'uploading' | 'parsing' | 'indexing' | 'ready' | 'error'
  summary?: string
  keywords?: string[]
  pageCount?: number
  metadata?: {
    stats?: Record<string, number>
    summary_preview?: string
  }
  uploadedAt: number
}

export interface LLMProvider {
  id: string
  name: string
  default_base_url: string
  models: string[]
}

export interface AppSettings {
  llm_provider: string
  llm_model: string
}

export interface Annotation {
  id: string
  paper_id: string
  page_number: number
  text_content: string
  color: string
  start_offset?: number
  end_offset?: number
  note?: string | null
  created_at?: string
}
