import { useCallback, useEffect, useRef, useState } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'
import AnnotationPopover from './AnnotationPopover'
import type { Annotation } from '../../types'

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`

function normalizeText(text: string): string {
  return text.replace(/\s+/g, ' ').trim()
}

interface PDFViewerProps {
  url: string
  annotations?: Annotation[]
  onAnnotation?: (color: string, text: string, pageNumber: number, note?: string) => void
  onTranslateSelection?: (text: string, pageNumber: number) => Promise<string>
  scrollToPage?: number
}

export default function PDFViewer({
  url,
  annotations = [],
  onAnnotation,
  onTranslateSelection,
  scrollToPage,
}: PDFViewerProps) {
  const [numPages, setNumPages] = useState<number>(0)
  const [scale, setScale] = useState(1.15)
  const [popover, setPopover] = useState<{
    x: number
    y: number
    text: string
    pageNumber: number
  } | null>(null)

  const containerRef = useRef<HTMLDivElement>(null)
  const pageRefs = useRef<Map<number, HTMLDivElement>>(new Map())
  const renderedPages = useRef<Set<number>>(new Set())

  const applyHighlights = useCallback((pageNumber: number) => {
    const pageEl = pageRefs.current.get(pageNumber)
    if (!pageEl) return

    const textLayer = pageEl.querySelector('.react-pdf__Page__textContent')
    if (!textLayer) return

    const spans = textLayer.querySelectorAll('span')
    const pageAnnotations = annotations.filter(a => a.page_number === pageNumber)

    spans.forEach(span => {
      span.style.backgroundColor = ''
      span.style.borderRadius = ''
      span.style.boxShadow = ''
    })

    if (pageAnnotations.length === 0) return

    const spanRanges: Array<{ span: HTMLSpanElement; start: number; end: number }> = []
    let offset = 0
    spans.forEach((span) => {
      const text = span.textContent || ''
      if (text.length > 0) {
        spanRanges.push({ span, start: offset, end: offset + text.length })
        offset += text.length
      }
    })

    const fullText = spanRanges.map(range => range.span.textContent || '').join('')
    const strippedChars: number[] = []
    for (let i = 0; i < fullText.length; i += 1) {
      if (!/\s/.test(fullText[i])) strippedChars.push(i)
    }
    const fullStripped = strippedChars.map(i => fullText[i]).join('')

    for (const ann of pageAnnotations) {
      const searchStripped = ann.text_content.replace(/\s/g, '')
      if (!searchStripped) continue

      const hex = ann.color.replace('#', '')
      const r = Number.parseInt(hex.substring(0, 2), 16)
      const g = Number.parseInt(hex.substring(2, 4), 16)
      const b = Number.parseInt(hex.substring(4, 6), 16)
      const bgColor = `rgba(${r}, ${g}, ${b}, 0.34)`

      const strippedIdx = fullStripped.indexOf(searchStripped)
      if (strippedIdx === -1) continue

      const strippedEnd = strippedIdx + searchStripped.length
      const rawStart = strippedChars[strippedIdx]
      const rawEnd = strippedEnd < strippedChars.length ? strippedChars[strippedEnd] : fullText.length

      for (const { span, start, end } of spanRanges) {
        if (end > rawStart && start < rawEnd) {
          span.style.backgroundColor = bgColor
          span.style.borderRadius = '6px'
          if (ann.note) {
            span.style.boxShadow = 'inset 0 -1px 0 rgba(31, 27, 22, 0.15)'
          }
        }
      }
    }
  }, [annotations])

  useEffect(() => {
    renderedPages.current.forEach(pageNum => applyHighlights(pageNum))
  }, [annotations, applyHighlights])

  const handlePageRenderSuccess = useCallback((pageNumber: number) => {
    renderedPages.current.add(pageNumber)
    applyHighlights(pageNumber)
  }, [applyHighlights])

  const handleMouseUp = useCallback(() => {
    const selection = window.getSelection()
    if (!selection || selection.isCollapsed || !selection.toString().trim()) return

    const text = normalizeText(selection.toString())
    if (!text) return

    const anchorNode = selection.anchorNode
    if (!anchorNode) return

    let pageElement: HTMLElement | null = anchorNode instanceof HTMLElement
      ? anchorNode
      : anchorNode.parentElement
    let pageNumber = 0

    while (pageElement) {
      const dataPage = pageElement.getAttribute('data-page-number')
      if (dataPage) {
        pageNumber = Number.parseInt(dataPage, 10)
        break
      }
      pageElement = pageElement.parentElement
    }

    if (!pageNumber) return

    const rect = selection.getRangeAt(0).getBoundingClientRect()
    const width = 320
    const x = Math.max(16, Math.min(window.innerWidth - width - 16, rect.left + rect.width / 2 - width / 2))
    const y = Math.max(16, rect.top - 20)

    setPopover({
      x,
      y,
      text,
      pageNumber,
    })
  }, [])

  const handleAnnotation = useCallback((color: string, text: string, pageNumber: number, note?: string) => {
    onAnnotation?.(color, text, pageNumber, note)
    setPopover(null)
    window.getSelection()?.removeAllRanges()
  }, [onAnnotation])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    const handleWheel = (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault()
        const delta = e.deltaY > 0 ? -0.05 : 0.05
        setScale(prev => Math.min(2.6, Math.max(0.65, prev + delta)))
      }
    }

    el.addEventListener('wheel', handleWheel, { passive: false })
    return () => el.removeEventListener('wheel', handleWheel)
  }, [])

  const scrollTo = useCallback((page: number) => {
    const el = pageRefs.current.get(page)
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [])

  useEffect(() => {
    if (!scrollToPage || scrollToPage <= 0) return
    const raf = requestAnimationFrame(() => scrollTo(scrollToPage))
    return () => cancelAnimationFrame(raf)
  }, [scrollToPage, scrollTo])

  return (
    <div className="h-full rounded-[28px] border border-border bg-surface shadow-[0_12px_40px_rgba(36,30,20,0.08)]">
      <div className="flex h-full flex-col overflow-hidden rounded-[28px]">
        <div className="flex h-14 items-center justify-between border-b border-border bg-[rgba(255,252,245,0.92)] px-4 backdrop-blur">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-text-tertiary">
              Reading Desk
            </p>
            <p className="text-sm text-text-secondary">
              选中文本后可自动翻译、高亮和做笔记
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setScale(prev => Math.max(0.65, prev - 0.1))}
              className="rounded-full border border-border bg-surface-secondary px-3 py-1.5 text-sm text-text-secondary transition-colors hover:bg-surface-tertiary"
            >
              缩小
            </button>
            <span className="w-16 text-center text-sm text-text-primary">
              {Math.round(scale * 100)}%
            </span>
            <button
              onClick={() => setScale(prev => Math.min(2.6, prev + 0.1))}
              className="rounded-full border border-border bg-surface-secondary px-3 py-1.5 text-sm text-text-secondary transition-colors hover:bg-surface-tertiary"
            >
              放大
            </button>
            <span className="rounded-full bg-surface-secondary px-3 py-1.5 text-xs text-text-secondary">
              {numPages > 0 ? `${numPages} 页` : '加载中'}
            </span>
          </div>
        </div>

        <div
          ref={containerRef}
          className="reader-canvas flex-1 overflow-auto px-4 py-5 md:px-8"
          onMouseUp={handleMouseUp}
        >
          <Document
            file={url}
            onLoadSuccess={({ numPages: totalPages }) => setNumPages(totalPages)}
            loading={
              <div className="flex h-full items-center justify-center">
                <p className="text-sm text-text-tertiary">正在加载论文 PDF...</p>
              </div>
            }
            error={
              <div className="flex h-full items-center justify-center">
                <p className="text-sm text-[#b1462f]">PDF 加载失败</p>
              </div>
            }
          >
            <div className="mx-auto flex max-w-[940px] flex-col items-center gap-6">
              {Array.from({ length: numPages }, (_, index) => (
                <div
                  key={index + 1}
                  ref={(el) => {
                    if (el) pageRefs.current.set(index + 1, el)
                  }}
                  className="relative"
                >
                  <div className="absolute -left-12 top-3 hidden rounded-full bg-surface px-3 py-1 text-xs text-text-tertiary shadow-sm md:block">
                    {index + 1}
                  </div>
                  <Page
                    pageNumber={index + 1}
                    scale={scale}
                    className="paper-sheet overflow-hidden rounded-[18px] shadow-[0_18px_48px_rgba(47,36,24,0.12)]"
                    renderTextLayer
                    renderAnnotationLayer
                    onRenderSuccess={() => handlePageRenderSuccess(index + 1)}
                  />
                </div>
              ))}
            </div>
          </Document>
        </div>

        {popover && (
          <AnnotationPopover
            x={popover.x}
            y={popover.y}
            selectedText={popover.text}
            pageNumber={popover.pageNumber}
            onHighlight={handleAnnotation}
            onTranslate={onTranslateSelection}
            onDismiss={() => setPopover(null)}
          />
        )}
      </div>
    </div>
  )
}
