export interface PaperSource {
  id: string
  type: string
  page_number: number
  title: string
  section?: string
  content: string
  score?: number
  asset_url?: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  sources?: PaperSource[]
}

export interface PaperFile {
  id: string
  name: string
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
