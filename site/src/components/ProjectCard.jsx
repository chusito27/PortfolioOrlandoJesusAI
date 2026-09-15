import { useState } from 'react'
import Chart from './Chart'

export default function ProjectCard({ project, activeTag, defaultOpen = false }) {
  // The first card opens by default: the measured results are the strongest
  // thing on the page, and they should not be behind a click.
  const [open, setOpen] = useState(defaultOpen)
  const detailId = `detail-${project.slug}`

  return (
    <article className="card">
      <div className="card-head">
        <div className="card-top">
          <h3>{project.title}</h3>
          <span className="badge">
            <span className="dot" />
            {project.status} · {project.year}
          </span>
        </div>

        <p className="card-tagline">{project.tagline}</p>

        {project.metrics?.length > 0 && (
          <div className="stats">
            {project.metrics.map((m) => (
              <div className="stat" key={m.label}>
                <div className="stat-value">{m.value}</div>
                <div className="stat-label">{m.label}</div>
                <div className="stat-note">{m.note}</div>
              </div>
            ))}
          </div>
        )}

        <div className="tags">
          {project.tech.map((t) => (
            <span className="tag" key={t} data-on={t === activeTag}>
              {t}
            </span>
          ))}
        </div>
      </div>

      {open && (
        <div className="detail" id={detailId}>
          <h4>What it does</h4>
          <p className="highlight-body">{project.summary}</p>

          {project.chart && (
            <>
              <h4>Measured results</h4>
              <Chart views={project.chart.views} />
            </>
          )}

          {project.highlights?.length > 0 && (
            <>
              <h4>Engineering decisions</h4>
              {project.highlights.map((h) => (
                <div className="highlight" key={h.title}>
                  <p className="highlight-title">{h.title}</p>
                  <p className="highlight-body">{h.body}</p>
                </div>
              ))}
            </>
          )}
        </div>
      )}

      <div className="card-actions">
        <button
          className="btn"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-controls={detailId}
        >
          {open ? 'Hide details' : 'Details and measurements'}
        </button>

        {project.links.repo && (
          <a className="btn btn-primary" href={project.links.repo} target="_blank" rel="noreferrer">
            View the code
          </a>
        )}

        {project.links.demo && (
          <a className="btn" href={project.links.demo} target="_blank" rel="noreferrer">
            Live demo
          </a>
        )}
      </div>
    </article>
  )
}
