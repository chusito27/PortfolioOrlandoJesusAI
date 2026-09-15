import { useState } from 'react'

/**
 * A horizontal bar figure with a tabbed view switch and a table fallback.
 *
 * One measure across a handful of categories, so a single hue carries the
 * magnitude and colour is never used to encode rank. The configuration that
 * actually shipped is called out in a second hue and named in the legend, so
 * that distinction never rests on colour alone.
 */
export default function Chart({ views }) {
  const [active, setActive] = useState(views[0].id)
  const [showTable, setShowTable] = useState(false)

  const view = views.find((v) => v.id === active) ?? views[0]
  const hasShipped = view.bars.some((b) => b.shipped)

  const format = (value) => `${value.toLocaleString()}${view.unit}`

  return (
    <div className="chart">
      {views.length > 1 && (
        <div className="chart-tabs" role="tablist" aria-label="Figure view">
          {views.map((v) => (
            <button
              key={v.id}
              role="tab"
              aria-selected={v.id === active}
              className="chart-tab"
              onClick={() => setActive(v.id)}
            >
              {v.label}
            </button>
          ))}
        </div>
      )}

      <div className="bars" role="img" aria-label={`${view.label}. ${view.caption}`}>
        {view.bars.map((bar) => (
          <div className="bar-row" key={bar.label} tabIndex={0}>
            <div className="bar-label">{bar.label}</div>
            <div className="bar-track">
              <div
                className="bar-fill"
                data-shipped={bar.shipped ? 'true' : 'false'}
                style={{ width: `${(bar.value / view.max) * 100}%` }}
              />
            </div>
            <div className="bar-value">{format(bar.value)}</div>
          </div>
        ))}
      </div>

      {hasShipped && (
        <div className="chart-legend">
          <span className="legend-item">
            <span className="swatch" style={{ background: 'var(--series-1)' }} />
            Measured alternative
          </span>
          <span className="legend-item">
            <span className="swatch" style={{ background: 'var(--series-shipped)' }} />
            Configuration that shipped
          </span>
        </div>
      )}

      <p className="chart-caption">{view.caption}</p>

      <button className="table-toggle" onClick={() => setShowTable((v) => !v)}>
        {showTable ? 'Hide data table' : 'Show data table'}
      </button>

      {showTable && (
        <table className="datatable">
          <caption className="sr-only">{view.label}</caption>
          <thead>
            <tr>
              <th scope="col">Configuration</th>
              <th scope="col">{view.label}</th>
            </tr>
          </thead>
          <tbody>
            {view.bars.map((bar) => (
              <tr key={bar.label}>
                <td>
                  {bar.label}
                  {bar.shipped ? ' (shipped)' : ''}
                </td>
                <td>{format(bar.value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
