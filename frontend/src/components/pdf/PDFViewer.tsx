import { useState, useCallback, useRef, useEffect } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'
import AnnotationPopover from './AnnotationPopover'
import type { Annotation } from '../../types'

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`

// Normalize whitespace: collapse runs of whitespace (including newlines) into single spaces, then trim
function normalizeText(text: string): string {
  return text.replace(/\s+/g, ' ').trim()
}

interface PDFViewerProps {
  url: string
  annotations?: Annotation[]
  onAnnotation?: (color: string, text: string, pageNumber: number) => void
  scrollToPage?: number
}

export default function PDFViewer({ url, annotations = [], onAnnotation, scrollToPage }: PDFViewerProps) {
  const [numPages, setNumPages] = useState<number>(0)
  const [scale, setScale] = useState(1.2)
  const [popover, setPopover] = useState<{
    x: number
    y: number
    text: string
    pageNumber: number
  } | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const pageRefs = useRef<Map<number, HTMLDivElement>>(new Map())
  const renderedPages = useRef<Set<number>>(new Set())

  // Apply annotation highlights to rendered PDF text layer spans
  const applyHighlights = useCallback((pageNumber: number) => {
    const pageEl = pageRefs.current.get(pageNumber)
    if (!pageEl) return

    const textLayer = pageEl.querySelector('.react-pdf__Page__textContent')
    if (!textLayer) return

    const spans = textLayer.querySelectorAll('span')
    const pageAnnotations = annotations.filter(a => a.page_number === pageNumber)

    // Clear previous highlights
    spans.forEach(span => {
      span.style.backgroundColor = ''
    })

    if (pageAnnotations.length === 0) return

    // Build raw span ranges — concatenate without separator to preserve exact character positions
    const spanRanges: Array<{ span: HTMLSpanElement; start: number; end: number }> = []
    let offset = 0
    spans.forEach(span => {
      const text = span.textContent || ''
      if (text.length > 0) {
        spanRanges.push({ span, start: offset, end: offset + text.length })
        offset += text.length
      }
    })
    const fullText = spanRanges.map(r => r.span.textContent || '').join('')

    // Build whitespace-stripped version + mapping back to raw positions
    const strippedChars: number[] = [] // strippedChars[strippedIdx] = rawIdx
    for (let i = 0; i < fullText.length; i++) {
      if (!/\s/.test(fullText[i])) {
        strippedChars.push(i)
      }
    }
    const fullStripped = strippedChars.map(i => fullText[i]).join('')

    // Apply highlights for each annotation
    for (const ann of pageAnnotations) {
      const searchStripped = ann.text_content.replace(/\s/g, '')
      if (!searchStripped) continue

      // Convert hex color to rgba with 0.3 opacity
      const hex = ann.color.replace('#', '')
      const r = parseInt(hex.substring(0, 2), 16)
      const g = parseInt(hex.substring(2, 4), 16)
      const b = parseInt(hex.substring(4, 6), 16)
      const bgColor = `rgba(${r}, ${g}, ${b}, 0.3)`

      // Find match in stripped text, then map back to raw positions
      const strippedIdx = fullStripped.indexOf(searchStripped)
      if (strippedIdx === -1) continue
      const strippedEnd = strippedIdx + searchStripped.length

      const rawStart = strippedChars[strippedIdx]
      const rawEnd = strippedEnd < strippedChars.length
        ? strippedChars[strippedEnd]
        : fullText.length

      // Highlight only the spans that overlap with this exact match range
      for (const { span, start, end } of spanRanges) {
        if (end > rawStart && start < rawEnd) {
          span.style.backgroundColor = bgColor
        }
      }
    }
  }, [annotations])

  // Re-apply highlights when annotations change
  useEffect(() => {
    renderedPages.current.forEach(pageNum => {
      applyHighlights(pageNum)
    })
  }, [annotations, applyHighlights])

  const handlePageRenderSuccess = useCallback((pageNumber: number) => {
    renderedPages.current.add(pageNumber)
    applyHighlights(pageNumber)
  }, [applyHighlights])

  // Detect text selection inside PDF pages
  const handleMouseUp = useCallback(() => {
    const selection = window.getSelection()
    if (!selection || selection.isCollapsed || !selection.toString().trim()) {
      return
    }

    const text = normalizeText(selection.toString())
    if (!text) return

    // Find which page the selection is in
    const anchorNode = selection.anchorNode
    if (!anchorNode) return

    let pageElement: HTMLElement | null = anchorNode instanceof HTMLElement
      ? anchorNode
      : anchorNode.parentElement
    let pageNumber = 0

    while (pageElement) {
      const dataPage = pageElement.getAttribute('data-page-number')
      if (dataPage) {
        pageNumber = parseInt(dataPage, 10)
        break
      }
      pageElement = pageElement.parentElement
    }

    if (pageNumber === 0) return

    // Get position for popover
    const range = selection.getRangeAt(0)
    const rect = range.getBoundingClientRect()

    setPopover({
      x: rect.left + rect.width / 2 - 75,
      y: rect.top - 45,
      text,
      pageNumber,
    })
  }, [])

  const handleHighlight = useCallback((color: string, text: string, pageNumber: number) => {
    onAnnotation?.(color, text, pageNumber)
    setPopover(null)
    window.getSelection()?.removeAllRanges()
  }, [onAnnotation])

  const dismissPopover = useCallback(() => {
    setPopover(null)
  }, [])

  // Ctrl+scroll / pinch-to-zoom
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const handleWheel = (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault()
        const delta = e.deltaY > 0 ? -0.05 : 0.05
        setScale(s => Math.min(3, Math.max(0.5, s + delta)))
      }
    }
    el.addEventListener('wheel', handleWheel, { passive: false })
    return () => el.removeEventListener('wheel', handleWheel)
  }, [])

  // Scroll to a specific page
  const scrollTo = useCallback((page: number) => {
    const el = pageRefs.current.get(page)
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [])

  // Handle scrollToPage prop changes
  if (scrollToPage && scrollToPage > 0) {
    // Use requestAnimationFrame to avoid calling during render
    requestAnimationFrame(() => scrollTo(scrollToPage))
  }

  return (
    <div className="h-full flex flex-col bg-surface-secondary">
      {/* Toolbar */}
      <div className="h-10 flex items-center justify-center gap-3 border-b border-border-light bg-surface px-3 shrink-0">
        <button
          onClick={() => setScale(s => Math.max(0.5, s - 0.1))}
          className="text-text-secondary hover:text-text-primary p-1 rounded hover:bg-surface-tertiary transition-colors text-sm"
        >
          −
        </button>
        <span className="text-xs text-text-secondary w-12 text-center">
          {Math.round(scale * 100)}%
        </span>
        <button
          onClick={() => setScale(s => Math.min(3, s + 0.1))}
          className="text-text-secondary hover:text-text-primary p-1 rounded hover:bg-surface-tertiary transition-colors text-sm"
        >
          +
        </button>
        <span className="text-xs text-text-tertiary mx-2">|</span>
        <span className="text-xs text-text-tertiary">
          {numPages > 0 ? `${numPages} pages` : ''}
        </span>
      </div>

      {/* PDF content */}
      <div
        ref={containerRef}
        className="flex-1 overflow-auto flex justify-center py-4"
        onMouseUp={handleMouseUp}
      >
        <Document
          file={url}
          onLoadSuccess={({ numPages: n }) => setNumPages(n)}
          loading={
            <div className="flex items-center justify-center h-full">
              <p className="text-sm text-text-tertiary">Loading PDF...</p>
            </div>
          }
          error={
            <div className="flex items-center justify-center h-full">
              <p className="text-sm text-red-500">Failed to load PDF</p>
            </div>
          }
        >
          <div className="flex flex-col items-center gap-4">
            {Array.from({ length: numPages }, (_, i) => (
              <div
                key={i + 1}
                ref={(el) => {
                  if (el) pageRefs.current.set(i + 1, el)
                }}
              >
                <Page
                  pageNumber={i + 1}
                  scale={scale}
                  className="shadow-sm"
                  renderTextLayer={true}
                  renderAnnotationLayer={true}
                  onRenderSuccess={() => handlePageRenderSuccess(i + 1)}
                />
              </div>
            ))}
          </div>
        </Document>
      </div>

      {/* Annotation color popover */}
      {popover && (
        <AnnotationPopover
          x={popover.x}
          y={popover.y}
          selectedText={popover.text}
          pageNumber={popover.pageNumber}
          onHighlight={handleHighlight}
          onDismiss={dismissPopover}
        />
      )}
    </div>
  )
}
