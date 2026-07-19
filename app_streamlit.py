import math, io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm, Normalize
from scipy.interpolate import griddata
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Deformación por Cizalla FL300 — Grupo 5",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header{background:#111;color:#fff;padding:18px 24px;border-radius:10px;margin-bottom:20px}
.main-header h1{font-size:20px;font-weight:700;margin:0 0 4px;color:#fff}
.main-header p{font-size:13px;color:rgba(255,255,255,.55);margin:0}
.header-badge{display:inline-block;background:#2480c8;color:#fff;font-size:12px;
  font-weight:600;padding:4px 14px;border-radius:20px;margin-top:10px}
[data-testid="stSidebar"]{background:#f0f6fc;border-right:1.5px solid #c8dff0}
[data-testid="metric-container"]{background:#e6f1fb;border:1px solid #b5d4f4;
  border-radius:10px;padding:10px 14px}
[data-testid="metric-container"] label{color:#185fa5!important;font-size:12px!important;font-weight:600!important}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#0c447c!important;font-size:22px!important;font-weight:700!important}
.section-header{background:#e6f1fb;border-left:4px solid #185fa5;padding:8px 14px;
  border-radius:0 8px 8px 0;margin:14px 0 10px;font-size:14px;font-weight:600;color:#0c447c}
.info-strip{background:#e6f1fb;border:1px solid #b5d4f4;border-radius:8px;
  padding:9px 14px;font-size:13px;color:#0c447c;margin-bottom:14px}
.step-card{background:#f0f6fc;border:1px solid #c8dff0;border-radius:10px;padding:14px;margin-bottom:10px}
.step-num{font-size:10px;font-weight:700;color:#2480c8;text-transform:uppercase;letter-spacing:.6px;margin-bottom:3px}
.step-title{font-size:14px;font-weight:700;color:#111;margin-bottom:7px}
.step-result{background:#fff;border:1px solid #b5d4f4;border-radius:7px;
  padding:9px 12px;font-family:monospace;font-size:13px;color:#111;margin-top:8px;line-height:1.8}
.val{color:#0c447c;font-weight:700}
.result-box{background:#111;border-radius:12px;padding:20px;text-align:center;margin-top:14px;color:#fff}
.result-box .rv{font-size:24px;font-weight:700;margin:6px 0}
.result-box .rs{font-size:13px;color:rgba(255,255,255,.6)}
</style>
""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────────────────────
KT   = 0.514444
DEG  = 110_000.0
DLAT = DLON = 5.0
LON0,LON1,LAT0,LAT1 = -132,-33,-28,48

# ── Datos ─────────────────────────────────────────────────────────────────────
@st.cache_data
def cargar():
    raw=[
        (45,-130,250,60),(45,-125,250,81),(45,-120,230,66),(45,-115,260,73),(45,-110,300,77),(45,-105,290,69),
        (40,-130,250,97),(40,-125,260,81),(40,-120,270,60),(40,-115,270,58),(40,-110,290,60),(40,-105,310,64),
        (35,-130,260,66),(35,-125,270,56),(35,-120,280,48),(35,-115,280,48),(35,-110,290,48),(35,-105,300,62),
        (45,-100,200,66),(45,-95,220,52),(45,-90,230,56),(45,-85,280,69),(45,-80,310,85),(45,-75,330,124),(45,-70,170,12),
        (40,-100,200,42),(40,-95,210,58),(40,-90,250,46),(40,-85,280,44),(40,-80,300,69),(40,-75,320,102),(40,-70,320,120),
        (35,-100,250,68),(35,-95,230,48),(35,-90,250,42),(35,-85,260,41),(35,-80,290,54),(35,-75,300,66),(35,-70,310,102),
        (30,-120,280,27),(30,-115,290,25),(30,-110,290,25),(30,-105,270,33),(30,-100,250,42),(30,-95,250,39),
        (25,-120,330,17),(25,-115,360,15),(25,-110,270,15),(25,-105,240,42),(25,-100,250,44),(25,-95,260,44),
        (20,-120,300,33),(20,-115,260,48),(20,-110,250,68),(20,-105,250,54),(20,-100,260,41),(20,-95,270,37),
        (15,-120,270,41),(15,-115,250,52),(15,-110,270,64),(15,-105,270,37),(15,-100,290,17),(15,-95,310,19),
        (30,-90,260,31),(30,-85,270,19),(30,-80,290,19),(30,-75,290,42),(30,-70,290,58),(30,-65,290,69),
        (25,-90,270,37),(25,-85,280,41),(25,-80,270,44),(25,-75,280,50),(25,-70,280,56),(25,-65,280,62),
        (20,-90,280,37),(20,-85,260,44),(20,-80,260,48),(20,-75,270,48),(20,-70,270,42),(20,-65,280,44),
        (15,-90,270,21),(15,-85,250,23),(15,-80,250,25),(15,-75,280,19),(15,-70,270,17),(15,-65,290,27),
        (10,-100,20,12),(10,-95,290,8),(10,-90,240,8),(10,-85,200,21),(10,-80,160,19),(10,-75,160,15),(10,-70,220,17),
        (5,-100,60,8),(5,-95,90,10),(5,-90,130,21),(5,-85,120,31),(5,-80,130,25),(5,-75,110,21),(5,-70,160,4),
        (0,-100,60,14),(0,-95,100,10),(0,-90,110,19),(0,-85,110,25),(0,-80,100,23),(0,-75,90,27),(0,-70,90,14),
        (10,-65,310,8),(10,-60,330,12),(10,-55,350,14),(10,-50,340,25),(10,-45,320,27),(10,-40,290,35),
        (5,-65,240,4),(5,-60,30,8),(5,-55,20,12),(5,-50,340,19),(5,-45,310,19),(5,-40,290,27),
        (0,-65,140,12),(0,-60,70,15),(0,-55,80,27),(0,-50,50,14),(0,-45,60,8),(0,-40,80,4),
        (-5,-85,170,8),(-5,-80,130,15),(-5,-75,100,21),(-5,-70,50,14),(-5,-65,50,17),
        (-10,-85,290,29),(-10,-80,270,15),(-10,-75,330,10),(-10,-70,20,15),(-10,-65,300,4),
        (-15,-85,300,42),(-15,-80,310,41),(-15,-75,290,33),(-15,-70,290,27),(-15,-65,270,27),
        (-20,-85,270,31),(-20,-80,300,41),(-20,-75,300,58),(-20,-70,290,52),(-20,-65,270,46),
        (-25,-85,260,27),(-25,-80,290,35),(-25,-75,290,44),(-25,-70,280,54),(-25,-65,270,58),
        (-5,-60,90,12),(-5,-55,50,12),(-5,-50,80,2),(-5,-45,60,6),(-5,-40,90,4),(-5,-35,330,6),
        (-10,-60,150,4),(-10,-55,250,10),(-10,-50,270,10),(-10,-45,30,4),(-10,-40,200,2),(-10,-35,90,2),
        (-15,-60,250,27),(-15,-55,250,23),(-15,-50,280,15),(-15,-45,280,23),(-15,-40,290,15),(-15,-35,330,4),
        (-20,-60,260,50),(-20,-55,260,50),(-20,-50,260,46),(-20,-45,290,42),(-20,-40,290,27),(-20,-35,290,25),
        (-25,-60,270,56),(-25,-55,270,50),(-25,-50,290,54),(-25,-45,290,50),(-25,-40,290,54),(-25,-35,290,39),
    ]
    def uv(d,s):
        m=s*KT; r=math.radians(d)
        return -m*math.sin(r),-m*math.cos(r)
    grid={}
    for lat,lon,d,s in raw:
        u,v=uv(d,s)
        grid[(lat,lon)]=dict(u=u,v=v,dir=d,spd_kt=s,spd_ms=s*KT)
    res,desc=[],[]
    for (lat,lon),pt in grid.items():
        vR=grid.get((lat,lon+DLON),{}).get('v')
        vL=grid.get((lat,lon-DLON),{}).get('v')
        uN=grid.get((lat+DLAT,lon),{}).get('u')
        uS=grid.get((lat-DLAT,lon),{}).get('u')
        if any(x is None for x in [vR,vL,uN,uS]):
            desc.append((lat,lon)); continue
        ciz=((vR-vL)/(2*DLON*DEG*math.cos(math.radians(lat)))+(uN-uS)/(2*DLAT*DEG))*1e5
        res.append(dict(lat=lat,lon=lon,dir=pt['dir'],spd_kt=pt['spd_kt'],
                        spd_ms=round(pt['spd_ms'],3),
                        u_ms=round(pt['u'],3),v_ms=round(pt['v'],3),
                        cizalla=round(ciz,4)))
    res.sort(key=lambda r:(-r['lat'],r['lon']))
    return grid,res,desc

grid,resultados,descartados=cargar()
vals=np.array([r['cizalla'] for r in resultados])

# ── Función de mapa con matplotlib puro (sin Cartopy) ─────────────────────────
@st.cache_data
def make_map(campo,titulo,label_cb,cmap_name,div,
             n_iso,alpha,sc_v,show_vec,show_iso,show_zero,show_brd,dpi):

    lats_i=np.array([r['lat']   for r in resultados])
    lons_i=np.array([r['lon']   for r in resultados])
    vals_i=np.array([r[campo]   for r in resultados])

    # Interpolación
    lon_g=np.linspace(LON0,LON1,220)
    lat_g=np.linspace(LAT0,LAT1,160)
    LON_G,LAT_G=np.meshgrid(lon_g,lat_g)
    FIELD=griddata((lons_i,lats_i),vals_i,(LON_G,LAT_G),method='linear')

    vmax=(max(abs(float(np.nanmin(FIELD))),abs(float(np.nanmax(FIELD))))
          if div else float(np.nanmax(FIELD)))
    norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax) if div else Normalize(0,vmax)
    cm=plt.get_cmap(cmap_name)

    fig,ax=plt.subplots(figsize=(14,7),dpi=dpi)
    fig.patch.set_facecolor('#f5f9fd')
    ax.set_facecolor('#d6eaf8')

    # Relleno
    cf=ax.contourf(LON_G,LAT_G,FIELD,levels=28,cmap=cm,norm=norm,alpha=alpha,extend='both')

    # Isolíneas
    if show_iso:
        cl=ax.contour(LON_G,LAT_G,FIELD,levels=n_iso,
                      colors='#1a1a2e',linewidths=0.5,alpha=0.6)
        ax.clabel(cl,fmt='%.1f',fontsize=7,inline=True)
    if show_zero and div:
        ax.contour(LON_G,LAT_G,FIELD,levels=[0],
                   colors='#111',linewidths=1.8,linestyles='--')

    # Líneas de referencia geográficas
    for lat_r,lbl,col in [(0,'Ecuador','#c0392b'),
                           (23.5,'T. Cáncer','#e74c3c'),
                           (-23.5,'T. Capricornio','#2471a3')]:
        ax.axhline(lat_r,color=col,lw=0.9,ls='--',alpha=0.75)
        ax.text(LON0+1,lat_r+0.8,lbl,fontsize=7.5,color=col)

    # Vectores de viento
    if show_vec:
        for r in resultados:
            uu=r['u_ms']*sc_v; vv=r['v_ms']*sc_v
            if abs(uu)+abs(vv)<0.03: continue
            ax.annotate('',
                xy=(r['lon']+uu,r['lat']+vv),
                xytext=(r['lon'],r['lat']),
                arrowprops=dict(arrowstyle='-|>',color='#0c447c',
                                lw=0.7,mutation_scale=7))

    # Puntos interiores
    ax.scatter(lons_i,lats_i,c=vals_i,cmap=cm,norm=norm,
               s=22,zorder=5,edgecolors='#0c447c',lw=0.4)

    # Bordes descartados
    if show_brd:
        lats_b=np.array([d[0] for d in descartados])
        lons_b=np.array([d[1] for d in descartados])
        ax.scatter(lons_b,lats_b,marker='X',color='#e67e22',
                   s=55,zorder=5,edgecolors='#8b4500',lw=0.8,
                   label=f'Borde descartado ({len(descartados)} pts)')
        ax.legend(loc='lower left',fontsize=8,framealpha=0.92,
                  facecolor='#f0f6fc',edgecolor='#c8dff0')

    # Cuadrícula y ejes
    ax.set_xlim(LON0,LON1); ax.set_ylim(LAT0,LAT1)
    ax.set_xlabel('Longitud (°W)',fontsize=9,color='#3a5570')
    ax.set_ylabel('Latitud (°N/S)',fontsize=9,color='#3a5570')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.grid(color='#5a7a9a',alpha=0.2,lw=0.4,ls='--')
    ax.tick_params(colors='#3a5570',labelsize=8)

    cb=fig.colorbar(cf,ax=ax,pad=0.01,shrink=0.82,aspect=26,
                    extend='both' if div else 'max')
    cb.set_label(label_cb,fontsize=9,labelpad=8,color='#0c447c')
    cb.ax.tick_params(labelsize=8)

    ax.set_title(titulo,fontsize=10,fontweight='bold',
                 pad=8,color='#0c447c')
    plt.tight_layout()
    buf=io.BytesIO()
    fig.savefig(buf,format='png',bbox_inches='tight',
                facecolor='#f5f9fd',dpi=dpi)
    plt.close(fig); buf.seek(0)
    return buf

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>Deformación por Cizalla Horizontal — FL300</h1>
  <p>d = ∂v/∂x + ∂u/∂y &nbsp;·&nbsp; Diferencias finitas centradas &nbsp;·&nbsp;
     1 kt = 0.5144 m/s &nbsp;·&nbsp; 1° = 110 km &nbsp;·&nbsp;
     Válido: 24-Ene-2026 12:00 UTC (07:00 hora Perú)</p>
  <span class="header-badge">Grupo 5 &nbsp;·&nbsp; FL300 &nbsp;·&nbsp; 30 000 ft</span>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">Parámetro calculado</div>',
                unsafe_allow_html=True)
    st.latex(r"d = \frac{\partial v}{\partial x} + \frac{\partial u}{\partial y}")
    st.caption("Diferencias finitas centradas · orden 2")
    st.divider()

    st.markdown('<div class="section-header">Controles del mapa</div>',
                unsafe_allow_html=True)
    show_vec  = st.checkbox("Vectores de viento", True)
    show_iso  = st.checkbox("Isolíneas de d", True)
    show_zero = st.checkbox("Isolínea d = 0", True)
    show_brd  = st.checkbox("Bordes descartados", True)
    n_iso  = st.slider("N° de isolíneas", 5, 22, 14)
    alpha  = st.slider("Opacidad relleno", 0.3, 1.0, 0.75, 0.05)
    sc_v   = st.slider("Escala vectores", 0.04, 0.22, 0.10, 0.01)
    dpi_sl = st.select_slider("Resolución DPI",
                               options=[80,100,120,150], value=100)
    st.divider()

    st.markdown('<div class="section-header">Estadísticas FL300</div>',
                unsafe_allow_html=True)
    c1,c2=st.columns(2)
    c1.metric("Mínimo d",  f"{vals.min():.3f}")
    c2.metric("Máximo d",  f"{vals.max():.3f}")
    c1.metric("Media d",   f"{vals.mean():.3f}")
    c2.metric("Sigma d",   f"{vals.std():.3f}")
    st.metric("Puntos calculados",  f"{len(resultados)}")
    st.metric("Bordes descartados", f"{len(descartados)}")
    st.caption("Unidades: ×10⁻⁵ s⁻¹")

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "Mapa de cizalla","Componentes u y v",
    "Ejemplo numérico","Código esencial","Tabla de datos"
])

# ── TAB 1 ──────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(
        '<div class="info-strip">Rojo = d &gt; 0 &nbsp;·&nbsp; '
        'Azul = d &lt; 0 &nbsp;·&nbsp; Línea discontinua = d = 0 &nbsp;·&nbsp; '
        'X naranja = borde descartado (60 pts)</div>',
        unsafe_allow_html=True)

    buf=make_map(
        campo='cizalla',
        titulo='d = ∂v/∂x + ∂u/∂y  |  Dif. finitas centradas  |  '
               f'{len(resultados)} pts calculados  |  {len(descartados)} bordes descartados',
        label_cb='Deformación por Cizalla (×10⁻⁵ s⁻¹)',
        cmap_name='RdBu_r', div=True,
        n_iso=n_iso, alpha=alpha, sc_v=sc_v,
        show_vec=show_vec, show_iso=show_iso,
        show_zero=show_zero, show_brd=show_brd, dpi=dpi_sl
    )
    st.image(buf, use_container_width=True)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Mínimo",  f"{vals.min():.4f} ×10⁻⁵ s⁻¹")
    c2.metric("Máximo",  f"{vals.max():.4f} ×10⁻⁵ s⁻¹")
    c3.metric("Media",   f"{vals.mean():.4f} ×10⁻⁵ s⁻¹")
    c4.metric("Sigma",   f"{vals.std():.4f} ×10⁻⁵ s⁻¹")

    buf_dl=make_map(
        campo='cizalla', titulo='',
        label_cb='d (×10⁻⁵ s⁻¹)', cmap_name='RdBu_r', div=True,
        n_iso=14, alpha=0.75, sc_v=0.10,
        show_vec=True, show_iso=True, show_zero=True,
        show_brd=True, dpi=150
    )
    st.download_button("Descargar mapa PNG (150 DPI)", data=buf_dl,
                       file_name="mapa_cizalla_FL300.png", mime="image/png")

# ── TAB 2 ──────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(
        '<div class="info-strip">u = −V·sin(θ) &nbsp;·&nbsp; v = −V·cos(θ) &nbsp;·&nbsp; '
        'Signo negativo: la dirección meteorológica indica de dónde viene el viento</div>',
        unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Componente zonal u (m/s)</div>',
                    unsafe_allow_html=True)
        bu=make_map(campo='u_ms',
            titulo='u = −V·sin(θ)  |  Componente zonal (m/s)',
            label_cb='u (m/s)', cmap_name='RdBu_r', div=True,
            n_iso=10, alpha=0.80, sc_v=0.09,
            show_vec=False, show_iso=False, show_zero=True,
            show_brd=False, dpi=100)
        st.image(bu, use_container_width=True)
        st.caption("Azul = flujo al Este  ·  Rojo = flujo al Oeste")
    with c2:
        st.markdown('<div class="section-header">Componente meridional v (m/s)</div>',
                    unsafe_allow_html=True)
        bv=make_map(campo='v_ms',
            titulo='v = −V·cos(θ)  |  Componente meridional (m/s)',
            label_cb='v (m/s)', cmap_name='BrBG', div=True,
            n_iso=10, alpha=0.80, sc_v=0.09,
            show_vec=False, show_iso=False, show_zero=True,
            show_brd=False, dpi=100)
        st.image(bv, use_container_width=True)
        st.caption("Azul = flujo al Norte  ·  Rojo = flujo al Sur")

    st.markdown('<div class="section-header">Velocidad escalar |V| (m/s)</div>',
                unsafe_allow_html=True)
    bs=make_map(campo='spd_ms',
        titulo='|V| = √(u²+v²)  |  Velocidad escalar (m/s)',
        label_cb='|V| (m/s)', cmap_name='YlOrRd', div=False,
        n_iso=10, alpha=0.80, sc_v=0.09,
        show_vec=False, show_iso=False, show_zero=False,
        show_brd=False, dpi=100)
    st.image(bs, use_container_width=True)
    st.caption("Amarillo = velocidad baja  ·  Rojo intenso = chorro subtropical (40°N)")

# ── TAB 3 ──────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(
        '<div class="info-strip">Nodo de máxima cizalla del dominio — '
        '40°N / 105°W &nbsp;·&nbsp; Código F300: <b>31064M46</b></div>',
        unsafe_allow_html=True)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Código WINTEM", "31064M46")
    c2.metric("Dirección",     "310°")
    c3.metric("Velocidad",     "64 kt")
    c4.metric("Temperatura",   "−46°C")

    st.markdown("""
    <div class="step-card">
      <div class="step-num">Paso 2</div>
      <div class="step-title">Convertir a m/s</div>
      <div class="step-result">V = 64 × 0.5144 = <span class="val">32.92 m/s</span></div>
    </div>
    <div class="step-card">
      <div class="step-num">Paso 3</div>
      <div class="step-title">Calcular u y v</div>
      <div class="step-result">
        u = −32.92 × sin(310°) = <span class="val">+25.22 m/s</span><br>
        v = −32.92 × cos(310°) = <span class="val">−21.16 m/s</span>
      </div>
    </div>
    <div class="step-card">
      <div class="step-num">Pasos 5 y 6</div>
      <div class="step-title">Distancias y diferencias finitas</div>
      <div class="step-result">
        2Δy = 1 100 000 m &nbsp;·&nbsp; 2Δx = 842 649 m<br>
        ∂v/∂x = (20.30 − (−10.56)) / 842 649 = <span class="val">3.663 × 10⁻⁵ s⁻¹</span><br>
        ∂u/∂y = (33.36 − 27.62) / 1 100 000 = <span class="val">0.521 × 10⁻⁵ s⁻¹</span>
      </div>
    </div>
    <div class="result-box">
      <div>Resultado final — Paso 7</div>
      <div class="rv">d = 3.663 + 0.521 = 4.18 × 10⁻⁵ s⁻¹</div>
      <div class="rs">Máximo del dominio &nbsp;·&nbsp; 40°N / 105°W &nbsp;·&nbsp; Chorro del Pacífico</div>
    </div>
    """, unsafe_allow_html=True)

# ── TAB 4 ──────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown(
        '<div class="info-strip">Solo las tres partes fundamentales del método — '
        'independientes de la lectura del archivo y de la visualización</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="section-header">Parte 1 — Conversión a u, v</div>',
                unsafe_allow_html=True)
    st.code("""KT_TO_MS = 0.514444   # 1 kt = 0.514444 m/s

def wind_to_uv(direction_deg, speed_kt):
    spd_ms = speed_kt * KT_TO_MS
    theta  = math.radians(direction_deg)
    u = -spd_ms * math.sin(theta)   # zonal   (Este+)
    v = -spd_ms * math.cos(theta)   # meridional (Norte+)
    return u, v""", language='python')

    st.markdown('<div class="section-header">Parte 2 — Distancias en metros</div>',
                unsafe_allow_html=True)
    st.code("""DEG_M = 110_000.0   # 1° = 110 km (indicación del profesor)
DLAT  = 5.0
DLON  = 5.0

def dos_dx(lat):
    return 2 * DLON * DEG_M * math.cos(math.radians(lat))

def dos_dy():
    return 2 * DLAT * DEG_M   # = 1 100 000 m (constante)""", language='python')

    st.markdown('<div class="section-header">Parte 3 — Diferencias finitas centradas</div>',
                unsafe_allow_html=True)
    st.code("""for (lat, lon), pt in grid.items():
    v_este  = grid.get((lat, lon+5), {}).get('v')
    v_oeste = grid.get((lat, lon-5), {}).get('v')
    u_norte = grid.get((lat+5, lon), {}).get('u')
    u_sur   = grid.get((lat-5, lon), {}).get('u')

    if any(x is None for x in [v_este, v_oeste, u_norte, u_sur]):
        continue   # borde → descartado (60 puntos)

    dvdx = (v_este  - v_oeste) / dos_dx(lat)   # ∂v/∂x
    dudy = (u_norte - u_sur)   / dos_dy()      # ∂u/∂y
    d    = (dvdx + dudy) * 1e5                 # ×10⁻⁵ s⁻¹""", language='python')

# ── TAB 5 ──────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Filtros</div>', unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    lat_f=c1.slider("Latitud", -25, 45, (-25,45))
    spd_f=c2.slider("Velocidad (kt)", 0, 130, (0,130))
    ciz_f=c3.slider("Cizalla (×10⁻⁵ s⁻¹)", -4.0, 4.5, (-4.0,4.5), 0.1)

    df=pd.DataFrame(resultados)
    df.columns=['Lat°N','Lon°W','Dir°','kt','m/s','u (m/s)','v (m/s)','d (×10⁻⁵ s⁻¹)']
    df['Lon°W']=df['Lon°W'].abs()
    mask=((df['Lat°N']>=lat_f[0])&(df['Lat°N']<=lat_f[1])&
          (df['kt']>=spd_f[0])&(df['kt']<=spd_f[1])&
          (df['d (×10⁻⁵ s⁻¹)']>=ciz_f[0])&(df['d (×10⁻⁵ s⁻¹)']<=ciz_f[1]))
    df_f=df[mask]
    st.caption(f"{len(df_f)} de {len(df)} puntos")
    st.dataframe(
        df_f.style.background_gradient(
            subset=['d (×10⁻⁵ s⁻¹)'], cmap='RdBu_r'),
        use_container_width=True, height=400)
    st.download_button("Descargar CSV", data=df_f.to_csv(index=False).encode(),
                       file_name="fl300_g5.csv", mime="text/csv")

st.divider()
st.caption("Grupo 5 · Meteorología Aeronáutica 2026 · "
           "WINTEM_2026-1_FL300_G5__1_.txt · "
           "d = ∂v/∂x + ∂u/∂y · 1 kt = 0.5144 m/s · 1° = 110 km")
