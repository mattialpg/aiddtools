(function () {
  const cfg = window.MOLDASH
  if (!cfg) return

  const clamp01 = x => (typeof x === 'number' ? Math.min(1, Math.max(0, x)) : 0)
  const clamp100 = x => (typeof x === 'number' ? Math.min(100, Math.max(0, x)) : 0)

  const topLabels = [
    { t: '\u2013\u2009\u2013\u2009\u2013', x: 4.5 },
    { t: '\u2013\u2009\u2013', x: 21 },
    { t: '\u2013', x: 41 },
    { t: '+', x: 60 },
    { t: '++', x: 80 },
    { t: '+++', x: 95.5 }
  ]

  const bottomLabels = [
    { t: '0', x: 0 },
    { t: '10', x: 10 },
    { t: '30', x: 30 },
    { t: '50', x: 50 },
    { t: '70', x: 70 },
    { t: '90', x: 90 },
    { t: '100', x: 100 }
  ]

  const ICON_COLLAPSED = 'admet-toggle-expand.png'
  const ICON_EXPANDED = 'admet-toggle-collapse.png'

  function makeWidget (ep) {
    // pull out fields with defaults; use unary + for numeric conversion below
    const {
      id: rawId = '',
      label: rawLabel,
      kind: rawKind = '',
      domain: epDomain,
      acceptance: epAcc,
      probability = 0,
      percentile = 0,
      value = 0
    } = ep || {}

    const id = String(rawId)
    const label = rawLabel ?? id

    const kind = String(rawKind)
    const isContinuous =
      kind === 'continuous' ||
      (epDomain && Number.isFinite(+epDomain.min) && Number.isFinite(+epDomain.max)) ||
      (epAcc && (Number.isFinite(+epAcc.min) || Number.isFinite(+epAcc.max)))

    const probPct = clamp01(+probability) * 100
    const pct = clamp100(+percentile)

    const wrap = document.createElement('div')
    wrap.className = 'admet-entry'

    if (isContinuous) {
      // default domain/acceptance values
      const { min: dminRaw = 0, max: dmaxRaw = 1 } = epDomain || {}
      const { min: accMinRaw, max: accMaxRaw } = epAcc || {}

      const dmin = +dminRaw
      const dmax = +dmaxRaw
      const span = dmax - dmin || 1

      const v = +value
      const xVal = Math.max(0, Math.min(1, (v - dmin) / span))
      const leftPct = (xVal * 100).toFixed(2)

      const accMinNum = +accMinRaw
      const accMaxNum = +accMaxRaw

      const hasMin = Number.isFinite(accMinNum)
      const hasMax = Number.isFinite(accMaxNum)

      let accMinLabel = dmin
      let accMaxLabel = dmax

      let accMinPct = 0
      let accMaxPct = 100

      if (hasMin || hasMax) {
        const amin = hasMin ? accMinNum : dmin
        const amax = hasMax ? accMaxNum : dmax

        accMinLabel = amin
        accMaxLabel = amax

        const amin01 = Math.max(0, Math.min(1, (amin - dmin) / span))
        const amax01 = Math.max(0, Math.min(1, (amax - dmin) / span))

        accMinPct = +((amin01 * 100).toFixed(2))
        accMaxPct = +((amax01 * 100).toFixed(2))
      }

      if (accMinPct > accMaxPct) {
        // swap using array destructuring for clarity
        [accMinPct, accMaxPct] = [accMaxPct, accMinPct]
        [accMinLabel, accMaxLabel] = [accMaxLabel, accMinLabel]
      }

      const leftOutside = accMinPct
      const rightOutside = 100 - accMaxPct

      const minRedBand = 4
      const minYellowBand = 2
      const edgeTightness = 0.85

      let leftRedPct = 0
      let leftMidPct = 0
      if (leftOutside > 0) {
        leftRedPct = Math.min(accMinPct, minRedBand)
        leftMidPct = Math.min(
          accMinPct,
          Math.max(leftRedPct + minYellowBand, accMinPct * edgeTightness)
        )
      }

      let rightRedPct = 100
      let rightMidPct = 100
      if (rightOutside > 0) {
        rightRedPct = Math.max(accMaxPct, 100 - minRedBand)
        rightMidPct = Math.max(
          accMaxPct,
          Math.min(
            rightRedPct - minYellowBand,
            accMaxPct + rightOutside * (1 - edgeTightness)
          )
        )
      }

      const dminLabel = Number.isFinite(dmin) ? String(dmin) : '0'
      const dmaxLabel = Number.isFinite(dmax) ? String(dmax) : '1'
      const aMinLabel = Number.isFinite(accMinLabel) ? String(accMinLabel) : dminLabel
      const aMaxLabel = Number.isFinite(accMaxLabel) ? String(accMaxLabel) : dmaxLabel

      wrap.innerHTML = `
        <div class="risk2-inline">
          <div class="risk2-inline-title">${label}</div>

          <div class="risk2-stage2">

            <div class="risk2-topscale risk2-topscale-continuous">
              <span style="left:0%">${dminLabel}</span>
              ${accMinPct > 0 ? `<span style="left:${accMinPct}%">${aMinLabel}</span>` : ''}
              ${accMaxPct < 100 ? `<span style="left:${accMaxPct}%">${aMaxLabel}</span>` : ''}
              <span style="left:100%">${dmaxLabel}</span>
            </div>

            <div class="risk2-track continuous-track"
                 style="--acc-min:${accMinPct}%; --acc-max:${accMaxPct}%; --left-red:${leftRedPct.toFixed(2)}%; --left-mid:${leftMidPct.toFixed(2)}%; --right-mid:${rightMidPct.toFixed(2)}%; --right-red:${rightRedPct.toFixed(2)}%;">
              <div class="risk2-barclip">
                <div class="range-gradient"></div>
                <div class="risk2-fill" style="left:calc(${leftPct}% - 6px); right:0;"></div>
              </div>

              <div class="risk2-percentile"
                   data-pct="${pct.toFixed(6)}"
                   style="left:${pct.toFixed(2)}%">
                <div class="risk2-percentile-label">
                  <div class="v">${pct.toFixed(2)}% (DB)</div>
                </div>
              </div>

              <div class="risk2-knob" style="left:${leftPct}%"></div>
            </div>

            <div class="risk2-bottomscale risk2-bottomscale-continuous">
              ${bottomLabels.map(o => `<span style="left:${o.x}%">${o.t}</span>`).join('')}
            </div>

          </div>
        </div>
      `

      return wrap
    }

    wrap.innerHTML = `
      <div class="risk2-inline">
        <div class="risk2-inline-title">${label}</div>

        <div class="risk2-stage2">
          <div class="risk2-topscale">
            ${topLabels.map(o => `<span class="z" style="left:${o.x}%">${o.t}</span>`).join('')}
          </div>

          <div class="risk2-track">
            <div class="risk2-barclip">
              <div class="risk2-gradient"></div>
              <div class="risk2-fill" style="left:calc(${probPct.toFixed(2)}% - 6px); right:0;"></div>
            </div>

            <div class="risk2-sep" style="left:10%"></div>
            <div class="risk2-sep" style="left:30%"></div>
            <div class="risk2-sep" style="left:50%"></div>
            <div class="risk2-sep" style="left:70%"></div>
            <div class="risk2-sep" style="left:90%"></div>

            <div class="risk2-percentile" data-pct="${pct.toFixed(6)}" style="left:${pct.toFixed(2)}%">
              <div class="risk2-percentile-label">
                <div class="v">${pct.toFixed(2)}% (DB)</div>
              </div>
            </div>

            <div class="risk2-knob" style="left:${probPct.toFixed(2)}%"></div>
          </div>

          <div class="risk2-bottomscale">
            ${bottomLabels.map(o => `<span style="left:${o.x}%">${o.t}</span>`).join('')}
          </div>
        </div>
      </div>
    `
    return wrap
  }

  function render () {
    const host = document.getElementById('admet')
    if (!host) return

    host.innerHTML = ''

    const controls = document.createElement('div')
    controls.className = 'admet-controls'

    const btn = document.createElement('button')
    btn.className = 'admet-toggle'
    btn.type = 'button'
    btn.setAttribute('aria-label', 'Toggle annotations')

    btn.innerHTML = `
      <img
        id="admet-toggle-icon"
        src="${ICON_COLLAPSED}"
        class="admet-toggle-icon"
        alt=""
        aria-hidden="true">
    `

    const stored = window.localStorage ? window.localStorage.getItem('admet_show_annotations') : null
    let showAnnotations = stored === '1'

    controls.appendChild(btn)

    // keep the toggle inside a full-width header container located inside
    // the visible panel content (preferably the .card) so both title and
    // controls render within the panel bounds whether expanded or not.
    const panel = document.getElementById('admet-panel')
    if (panel) {
      // prefer inserting header inside the panel's .card if present
      const targetContainer = panel.querySelector('.card') || panel

      // ensure header wrapper exists at the top of the chosen container
      let header = targetContainer.querySelector('.admet-header')
      if (!header) {
        header = document.createElement('div')
        header.className = 'admet-header'
        targetContainer.insertBefore(header, targetContainer.firstChild)
      }

      // try to find a title already inside the panel; if none exists,
      // fall back to a global title text (if available) or create a default
      let titleEl = panel.querySelector('.admet-title, h1, h2, .title, [data-title]')
      if (!titleEl) {
        const globalTitle = document.querySelector('.admet-title, h1, h2, .title, [data-title]')
        const titleText = globalTitle ? (globalTitle.textContent || '').trim() : 'ADMET'
        const newTitle = document.createElement('div')
        newTitle.className = 'admet-title'
        newTitle.textContent = titleText
        header.appendChild(newTitle)
      } else if (titleEl.parentNode !== header) {
        // move the panel-local title into the header container
        header.appendChild(titleEl)
      }

      // append controls to the header; CSS will push them to the right edge
      header.appendChild(controls)
    } else {
      // fallback: if panel is missing, append controls to the host
      host.appendChild(controls)
    }

    const content = document.createElement('div')
    content.className = 'admet-content'
    host.appendChild(content)

    function snapPercentileMarkers () {
      const stages = content.querySelectorAll('.risk2-stage2')
      stages.forEach(function (stage) {
        const track = stage.querySelector('.risk2-track')
        const marker = stage.querySelector('.risk2-percentile')
        if (!track || !marker) return

        const pct = Number(marker.getAttribute('data-pct') || '0')
        const w = track.clientWidth
        const x = Math.round((pct / 100) * w)

        marker.style.left = x + 'px'
      })
    }

    function drawEndpoints () {
      content.innerHTML = ''

      const eps = Array.isArray(cfg.endpoints) ? cfg.endpoints : []

      if (!showAnnotations) {
        const medchem = eps.filter(function (ep) {
          const c = ep.category == null ? '' : String(ep.category).trim()
          return c === 'Medicinal Chemistry'
        })

        if (medchem.length) {
          medchem.forEach(function (ep) {
            content.appendChild(makeWidget(ep))
          })
          requestAnimationFrame(snapPercentileMarkers)
          return
        }

        const qed = eps.find(function (ep) {
          const rawId = String(ep.id || '')
          const rawLabel = ep.label == null ? '' : String(ep.label)
          const id = rawId
          const label = rawLabel
          return id === 'qed' || label.includes('qed')
        })

        if (qed) content.appendChild(makeWidget(qed))
        requestAnimationFrame(snapPercentileMarkers)
        return
      }

      const order = [
        'Medicinal Chemistry',
        'Physicochemical Properties',
        'Absorption',
        'Distribution',
        'Metabolism',
        'Excretion',
        'Toxicity',
        'Tox21 Pathways'
      ]

      const groups = new Map()
      order.forEach(k => groups.set(k, []))

      eps.forEach(function (ep) {
        const raw = ep.category == null ? '' : String(ep.category)
        const k = raw && raw.trim() ? raw.trim() : 'Physicochemical Properties'
        if (!groups.has(k)) groups.set(k, [])
        groups.get(k).push(ep)
      })

      order.forEach(function (k) {
        const items = groups.get(k) || []
        if (!items.length) return

        const title = document.createElement('div')
        title.className = 'admet-group-title'
        title.textContent = k
        content.appendChild(title)

        const grid = document.createElement('div')
        grid.className = 'admet-group-grid'
        content.appendChild(grid)

        items.forEach(function (ep) {
          grid.appendChild(makeWidget(ep))
        })
      })

      requestAnimationFrame(snapPercentileMarkers)
    }

    function sync () {
      btn.classList.toggle('active', showAnnotations)

      host.classList.toggle('admet-compact', !showAnnotations)

      const panel = document.getElementById('admet-panel')
      if (panel) panel.classList.toggle('admet-expanded', showAnnotations)

      const icon = btn.querySelector('#admet-toggle-icon')
      if (icon) icon.src = showAnnotations ? ICON_EXPANDED : ICON_COLLAPSED

      if (window.localStorage) {
        window.localStorage.setItem('admet_show_annotations', showAnnotations ? '1' : '0')
      }

      drawEndpoints()
    }

    btn.addEventListener('click', function () {
      showAnnotations = !showAnnotations
      sync()
    })

    sync()
  }

  render()
})()