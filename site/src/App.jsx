import { useEffect, useMemo, useState } from 'react'
import { profile, projects } from './data/projects'
import ProjectCard from './components/ProjectCard'

/** Theme is remembered per viewer, and every access is guarded: site data can be
 *  blocked, and a portfolio that throws on load is worse than one without a toggle. */
function readTheme() {
  try {
    return localStorage.getItem('theme')
  } catch {
    return null
  }
}

function useTheme() {
  const [theme, setTheme] = useState(readTheme)

  useEffect(() => {
    const root = document.documentElement
    if (theme) {
      root.setAttribute('data-theme', theme)
    } else {
      root.removeAttribute('data-theme')
    }
    try {
      if (theme) localStorage.setItem('theme', theme)
    } catch {
      /* private window or blocked storage: the toggle still works for this visit */
    }
  }, [theme])

  const resolved =
    theme ??
    (window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')

  return [resolved, () => setTheme(resolved === 'dark' ? 'light' : 'dark')]
}

export default function App() {
  const [theme, toggleTheme] = useTheme()
  const [activeTag, setActiveTag] = useState(null)

  const tags = useMemo(
    () => [...new Set(projects.flatMap((p) => p.tech))].sort((a, b) => a.localeCompare(b)),
    [],
  )

  const visible = activeTag ? projects.filter((p) => p.tech.includes(activeTag)) : projects

  return (
    <>
      <header className="topbar">
        <div className="wrap topbar-inner">
          <span className="mark">
            orlando<span>.</span>calvo
          </span>
          <nav className="topbar-links">
            <a
              className="iconbtn"
              href={profile.links.github}
              target="_blank"
              rel="noreferrer"
              aria-label="GitHub"
              title="GitHub"
            >
              <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <path d="M12 .5C5.37.5 0 5.87 0 12.5c0 5.3 3.44 9.8 8.21 11.39.6.11.82-.26.82-.58v-2.03c-3.34.73-4.04-1.61-4.04-1.61-.55-1.39-1.34-1.76-1.34-1.76-1.09-.75.08-.73.08-.73 1.2.09 1.84 1.24 1.84 1.24 1.07 1.84 2.81 1.31 3.5 1 .11-.78.42-1.31.76-1.61-2.67-.3-5.47-1.34-5.47-5.96 0-1.32.47-2.39 1.24-3.23-.13-.3-.54-1.53.11-3.18 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6.01 0c2.29-1.55 3.3-1.23 3.3-1.23.65 1.65.24 2.88.12 3.18.77.84 1.23 1.91 1.23 3.23 0 4.63-2.81 5.65-5.49 5.95.43.37.82 1.1.82 2.22v3.29c0 .32.21.7.83.58A12.01 12.01 0 0 0 24 12.5C24 5.87 18.63.5 12 .5z" />
              </svg>
            </a>
            <a
              className="iconbtn"
              href={profile.links.linkedin}
              target="_blank"
              rel="noreferrer"
              aria-label="LinkedIn"
              title="LinkedIn"
            >
              <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <path d="M4.98 3.5a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5zM2.4 21.5h5.16V9.06H2.4V21.5zM9.94 9.06h4.95v1.7h.07c.69-1.24 2.38-2.55 4.9-2.55 5.24 0 6.21 3.3 6.21 7.6v5.69h-5.16v-5.04c0-1.2-.02-2.75-1.73-2.75-1.73 0-2 1.31-2 2.66v5.13H9.94V9.06z" />
              </svg>
            </a>
            <a
              className="iconbtn"
              href={profile.links.email}
              aria-label="Email"
              title="Email"
            >
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                <rect x="2" y="4" width="20" height="16" rx="2" />
                <path d="m2 7 10 6 10-6" />
              </svg>
            </a>
            <button
              className="iconbtn"
              onClick={toggleTheme}
              aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
              title="Toggle theme"
            >
              {theme === 'dark' ? (
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                  <circle cx="12" cy="12" r="4.5" />
                  <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4" />
                </svg>
              ) : (
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                  <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
                </svg>
              )}
            </button>
          </nav>
        </div>
      </header>

      <main>
        <section className="hero">
          <div className="wrap">
            <p className="eyebrow">{profile.role}</p>
            <h1>{profile.name}</h1>
            <p className="hero-tagline">{profile.tagline}</p>
            <p className="hero-blurb">{profile.blurb}</p>
            <div className="facts">
              <span className="fact">
                <span className="dot" />
                Open to work
              </span>
              <span className="fact">
                <b>{profile.location}</b>
              </span>
              <span className="fact">{profile.timezone}</span>
              <span className="fact">
                English <b>{profile.english}</b>
              </span>
            </div>
          </div>
        </section>

        <section className="section">
          <div className="wrap">
            <div className="section-head">
              <h2>Projects</h2>
              <span className="count">
                {visible.length} of {projects.length} shown
              </span>
            </div>

            <div className="filters">
              <button
                className="chip"
                aria-pressed={activeTag === null}
                onClick={() => setActiveTag(null)}
              >
                All
              </button>
              {tags.map((tag) => (
                <button
                  key={tag}
                  className="chip"
                  aria-pressed={activeTag === tag}
                  onClick={() => setActiveTag(activeTag === tag ? null : tag)}
                >
                  {tag}
                </button>
              ))}
            </div>

            {visible.length === 0 ? (
              <p className="empty">Nothing tagged {activeTag} yet.</p>
            ) : (
              visible.map((p, i) => (
                <ProjectCard key={p.slug} project={p} activeTag={activeTag} defaultOpen={i === 0} />
              ))
            )}
          </div>
        </section>
      </main>

      <footer className="footer">
        <div
          className="wrap"
          style={{ display: 'flex', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap', width: '100%' }}
        >
          <span>Built with React and Vite. Every number on this page is reproducible from its repository.</span>
          <span>{profile.name}</span>
        </div>
      </footer>
    </>
  )
}
