const API_BASE = '/api'

export async function uploadPaper(file: File): Promise<{ id: string; filename: string; status: string }> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/papers/upload`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`)
  return res.json()
}

export async function getPaperStatus(paperId: string): Promise<{ id: string; status: string; summary?: string; keywords?: string }> {
  const res = await fetch(`${API_BASE}/papers/${paperId}/status`)
  if (!res.ok) throw new Error(`Status check failed: ${res.statusText}`)
  return res.json()
}

export async function createConversation(paperId: string): Promise<{ id: string; paper_id: string }> {
  const res = await fetch(`${API_BASE}/conversations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paper_id: paperId }),
  })
  if (!res.ok) throw new Error(`Create conversation failed: ${res.statusText}`)
  return res.json()
}

export interface StreamCallbacks {
  onRoute?: (route: string) => void
  onToken: (token: string) => void
  onDone: (fullAnswer: string) => void
  onError?: (error: string) => void
}

export function streamChat(
  conversationId: string,
  paperId: string,
  content: string,
  callbacks: StreamCallbacks,
): AbortController {
  const controller = new AbortController()

  fetch(`${API_BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      conversation_id: conversationId,
      paper_id: paperId,
      content,
    }),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        callbacks.onError?.(`Chat failed: ${res.statusText}`)
        return
      }

      const reader = res.body?.getReader()
      if (!reader) return

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            continue
          }
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6)
            try {
              const data = JSON.parse(dataStr)
              // Determine event type from the data content
              if (data.route !== undefined) {
                callbacks.onRoute?.(data.route)
              } else if (data.content !== undefined) {
                callbacks.onToken(data.content)
              } else if (data.answer !== undefined) {
                callbacks.onDone(data.answer)
              } else if (data.error !== undefined) {
                callbacks.onError?.(data.error)
              }
            } catch {
              // Skip non-JSON lines
            }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== 'AbortError') {
        callbacks.onError?.(err.message)
      }
    })

  return controller
}

export async function fetchPapers() {
  const res = await fetch(`${API_BASE}/papers`)
  if (!res.ok) throw new Error('Failed to fetch papers')
  return res.json()
}

export async function fetchPaperDetail(paperId: string): Promise<{
  id: string; filename: string; status: string; conversations: Array<{ id: string; title?: string; created_at: string }>
}> {
  const res = await fetch(`${API_BASE}/papers/${paperId}`)
  if (!res.ok) throw new Error('Failed to fetch paper detail')
  return res.json()
}

export async function fetchMessages(conversationId: string): Promise<Array<{
  id: string; role: string; content: string; created_at: string
}>> {
  const res = await fetch(`${API_BASE}/conversations/${conversationId}/messages`)
  if (!res.ok) throw new Error('Failed to fetch messages')
  return res.json()
}

export async function deletePaper(paperId: string) {
  const res = await fetch(`${API_BASE}/papers/${paperId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Failed to delete paper')
}

export async function fetchAnnotations(paperId: string) {
  const res = await fetch(`${API_BASE}/papers/${paperId}/annotations`)
  if (!res.ok) throw new Error('Failed to fetch annotations')
  return res.json()
}

export async function createAnnotation(paperId: string, data: {
  page_number: number; text_content: string; color: string;
  start_offset?: number; end_offset?: number;
}) {
  const res = await fetch(`${API_BASE}/papers/${paperId}/annotations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error('Failed to create annotation')
  return res.json()
}

export async function deleteAnnotation(paperId: string, annotationId: string) {
  const res = await fetch(`${API_BASE}/papers/${paperId}/annotations/${annotationId}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error('Failed to delete annotation')
}

export async function getProviders(): Promise<{ providers: Array<{ id: string; name: string; default_base_url: string; models: string[] }> }> {
  const res = await fetch(`${API_BASE}/settings/providers`)
  if (!res.ok) throw new Error('Failed to get providers')
  return res.json()
}

export async function updateSettings(data: {
  llm_provider?: string
  api_key?: string
  base_url?: string
  model?: string
}): Promise<void> {
  const res = await fetch(`${API_BASE}/settings`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error('Failed to update settings')
}