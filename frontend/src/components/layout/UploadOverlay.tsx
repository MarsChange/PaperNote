import { useCallback, useState } from 'react'

interface UploadOverlayProps {
  onFileSelect: (file: File) => void
  onOpenSettings: () => void
  onOpenHistory: () => void
}

const featureCards = [
  { title: 'MinerU 结构解析', description: '提取正文、图片、表格与公式，保留论文结构语义。' },
  { title: '混合检索问答', description: '基于向量 + 关键词检索，返回可定位的证据片段。' },
  { title: '沉浸式阅读', description: '选中段落即可翻译、高亮、记录问题与想法。' },
]

export default function UploadOverlay({ onFileSelect, onOpenSettings, onOpenHistory }: UploadOverlayProps) {
  const [isDragging, setIsDragging] = useState(false)

  const handleFile = useCallback((file: File) => {
    if (file.type === 'application/pdf') onFileSelect(file)
  }, [onFileSelect])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  return (
    <div
      className="relative flex h-full w-full items-center justify-center overflow-hidden px-4 py-8 md:px-8"
      onDragOver={(e) => {
        e.preventDefault()
        setIsDragging(true)
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
    >
      <div className="hero-orb hero-orb-left" />
      <div className="hero-orb hero-orb-right" />

      <div className="absolute left-4 top-4 flex gap-2 md:left-8 md:top-8">
        <button
          onClick={onOpenHistory}
          className="rounded-full border border-border bg-[rgba(255,252,245,0.84)] px-4 py-2 text-sm text-text-secondary shadow-sm backdrop-blur transition-colors hover:bg-surface"
        >
          文库历史
        </button>
        <button
          onClick={onOpenSettings}
          className="rounded-full border border-border bg-[rgba(255,252,245,0.84)] px-4 py-2 text-sm text-text-secondary shadow-sm backdrop-blur transition-colors hover:bg-surface"
        >
          模型设置
        </button>
      </div>

      <div className="grid w-full max-w-6xl gap-8 lg:grid-cols-[1.05fr_0.95fr]">
        <section className="animate-rise-in">
          <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-text-tertiary">
            PaperNote
          </p>
          <h1 className="mt-4 max-w-2xl font-display text-4xl leading-tight text-text-primary md:text-6xl">
            面向论文阅读的
            <span className="block text-accent">多模态检索智能体</span>
          </h1>
          <p className="mt-5 max-w-xl text-base leading-8 text-text-secondary md:text-lg">
            上传一篇 PDF，系统会自动完成结构解析、证据索引、阅读问答和笔记收集，
            用接近研究助手的方式帮你真正读懂论文。
          </p>

          <div className="mt-8 grid gap-4 md:grid-cols-3">
            {featureCards.map((card, index) => (
              <div
                key={card.title}
                className="animate-rise-in rounded-[28px] border border-border bg-[rgba(255,252,245,0.78)] p-5 shadow-[0_18px_45px_rgba(36,30,20,0.08)] backdrop-blur"
                style={{ animationDelay: `${index * 90}ms` }}
              >
                <p className="text-sm font-semibold text-text-primary">{card.title}</p>
                <p className="mt-2 text-sm leading-6 text-text-secondary">{card.description}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="animate-rise-in rounded-[32px] border border-border bg-[rgba(255,252,245,0.82)] p-5 shadow-[0_28px_80px_rgba(36,30,20,0.12)] backdrop-blur md:p-7">
          <div className="rounded-[28px] border border-dashed border-border bg-surface-secondary p-4">
            <div className="rounded-[24px] bg-surface px-5 py-6 shadow-sm">
              <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-text-tertiary">
                Upload PDF
              </p>
              <h2 className="mt-3 text-2xl font-semibold text-text-primary">
                开始一场新的论文阅读
              </h2>
              <p className="mt-2 text-sm leading-6 text-text-secondary">
                支持拖拽上传。解析完成后会自动进入阅读智能体工作台，并创建可追溯证据的问答会话。
              </p>

              <label
                className={`mt-6 flex min-h-[260px] cursor-pointer flex-col items-center justify-center rounded-[28px] border-2 border-dashed px-6 text-center transition-all ${
                  isDragging
                    ? 'border-accent bg-[rgba(15,118,110,0.08)] scale-[1.01]'
                    : 'border-border bg-surface-secondary hover:border-text-tertiary hover:bg-surface-tertiary'
                }`}
              >
                <div className="rounded-full bg-surface p-4 shadow-sm">
                  <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" className="text-accent">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                    <line x1="12" y1="18" x2="12" y2="10" />
                    <polyline points="9 13 12 10 15 13" />
                  </svg>
                </div>

                <p className="mt-5 text-xl font-semibold text-text-primary">
                  {isDragging ? '释放鼠标以上传 PDF' : '拖入 PDF 或点击这里选择文件'}
                </p>
                <p className="mt-2 max-w-sm text-sm leading-6 text-text-secondary">
                  推荐上传学术论文 PDF。系统会提取正文、图表和公式，并生成面向问答的多模态索引。
                </p>

                <div className="mt-6 rounded-full bg-text-primary px-5 py-2 text-sm font-medium text-surface">
                  选择论文文件
                </div>

                <input
                  type="file"
                  accept=".pdf"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (file) handleFile(file)
                  }}
                />
              </label>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
