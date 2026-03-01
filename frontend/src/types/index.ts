export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
}

export interface PaperFile {
  id: string
  name: string
  status: 'uploading' | 'parsing' | 'ready' | 'error'
  summary?: string
  keywords?: string[]
  uploadedAt: number
}

export interface LLMProvider {
  id: string
  name: string
  baseUrl: string
  models: string[]
}

export interface AppSettings {
  llmProvider: string
  llmModel: string
  apiKey: string
  baseUrl: string
}

export interface Annotation {
  id: string
  paper_id: string
  page_number: number
  text_content: string
  color: string
  start_offset?: number
  end_offset?: number
  created_at?: string
}