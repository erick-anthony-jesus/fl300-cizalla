# =============================================================================
#  GRUPO 5 — Meteorología Aeronáutica 2026
#  App Streamlit: Explorador interactivo FL300
#  Colores: azul oscuro (#0c447c) header + azul celeste (#e6f1fb) fondos
#
#  Instalar: pip install streamlit matplotlib scipy cartopy numpy
#  Correr:   streamlit run app_streamlit.py
# =============================================================================

import re, math, io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import streamlit as st

# ── Configuración ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deformación por Cizalla FL300 — Grupo 5",
    page_icon="🌬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Header principal */
.main-header {
    background: #111111;
    color: white;
    padding: 18px 24px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.main-header h1 {
    font-size: 20px;
    font-weight: 700;
    margin: 0 0 4px 0;
    color: white;
}
.main-header p {
    font-size: 13px;
    color: rgba(255,255,255,0.55);
    margin: 0;
}
.header-badge {
    display: inline-block;
    background: #2480c8;
    color: white;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 14px;
    border-radius: 20px;
    margin-top: 10px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f0f6fc;
    border-right: 1.5px solid #c8dff0;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.2rem;
}

/* Métricas */
[data-testid="metric-container"] {
    background: #e6f1fb;
    border: 1px solid #b5d4f4;
    border-radius: 10px;
    padding: 10px 14px;
}
[data-testid="metric-container"] label {
    color: #185fa5 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0c447c !important;
    font-size: 22px !important;
    font-weight: 700 !important;
}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    background: #f0f6fc;
    border-radius: 8px;
    padding: 4px;
    border: 1px solid #c8dff0;
}
[data-testid="stTabs"] [role="tab"] {
    color: #3a5570;
    font-weight: 500;
    font-size: 13px;
    border-radius: 6px;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #0c447c;
    color: white;
}

/* Bloques de código */
[data-testid="stCode"] {
    background: #f5f9fd;
    border: 1px solid #c8dff0;
    border-radius: 8px;
}

/* Botones */
[data-testid="stDownloadButton"] button {
    background: #0c447c;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 18px;
}
[data-testid="stDownloadButton"] button:hover {
    background: #185fa5;
}

/* Checkboxes y sliders */
[data-testid="stCheckbox"] label {
    color: #0c447c;
    font-weight: 500;
    font-size: 13px;
}
[data-testid="stSlider"] [role="slider"] {
    background: #2480c8;
}

/* Separador de sección */
.section-header {
    background: #e6f1fb;
    border-left: 4px solid #185fa5;
    padding: 8px 14px;
    border-radius: 0 8px 8px 0;
    margin: 14px 0 10px 0;
    font-size: 14px;
    font-weight: 600;
    color: #0c447c;
}

/* Info strip */
.info-strip {
    background: #e6f1fb;
    border: 1px solid #b5d4f4;
    border-radius: 8px;
    padding: 9px 14px;
    font-size: 13px;
    color: #0c447c;
    margin-bottom: 14px;
}

/* Step cards */
.step-card {
    background: #f0f6fc;
    border: 1px solid #c8dff0;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 10px;
}
.step-num {
    font-size: 10px;
    font-weight: 700;
    color: #2480c8;
    text-transform: uppercase;
    letter-spacing: .6px;
    margin-bottom: 3px;
}
.step-title {
    font-size: 14px;
    font-weight: 700;
    color: #111;
    margin-bottom: 7px;
}
.step-result {
    background: white;
    border: 1px solid #b5d4f4;
    border-radius: 7px;
    padding: 9px 12px;
    font-family: monospace;
    font-size: 13px;
    color: #111;
    margin-top: 8px;
    line-height: 1.8;
}
.val { color: #0c447c; font-weight: 700; }

/* Result box final */
.result-box {
    background: #111;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-top: 14px;
    color: white;
}
.result-box .rv {
    font-size: 24px;
    font-weight: 700;
    margin: 6px 0;
}
.result-box .rs {
    font-size: 13px;
    color: rgba(255,255,255,.6);
}

/* Tabla */
[data-testid="stDataFrame"] {
    border: 1px solid #c8dff0;
    border-radius: 10px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ── Constantes ─────────────────────────────────────────────────────────────────
KT_TO_MS  = 0.514444
DEG_M     = 110_000.0
DLAT = DLON = 5.0

# ── Datos ──────────────────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    raw = [
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
    def uv(d, s):
        m = s * KT_TO_MS
        r = math.radians(d)
        return -m*math.sin(r), -m*math.cos(r)

    grid = {}
    for lat,lon,dir_,spd in raw:
        u,v = uv(dir_, spd)
        grid[(lat,lon)] = dict(u=u,v=v,dir=dir_,spd_kt=spd,spd_ms=spd*KT_TO_MS)

    resultados, descartados = [], []
    for (lat,lon),pt in grid.items():
        vR=grid.get((lat,lon+DLON),{}).get('v')
        vL=grid.get((lat,lon-DLON),{}).get('v')
        uT=grid.get((lat+DLAT,lon),{}).get('u')
        uB=grid.get((lat-DLAT,lon),{}).get('u')
        if any(x is None for x in [vR,vL,uT,uB]):
            descartados.append((lat,lon)); continue
        dos_dx = 2*DLON*DEG_M*math.cos(math.radians(lat))
        dos_dy = 2*DLAT*DEG_M
        d = ((vR-vL)/dos_dx + (uT-uB)/dos_dy)*1e5
        resultados.append(dict(lat=lat,lon=lon,dir=pt['dir'],spd_kt=pt['spd_kt'],
                               spd_ms=round(pt['spd_ms'],3),
                               u_ms=round(pt['u'],3),v_ms=round(pt['v'],3),
                               cizalla=round(d,4)))
    resultados.sort(key=lambda r: (-r['lat'],r['lon']))
    return grid, resultados, descartados

grid, resultados, descartados = cargar_datos()
vals = np.array([r['cizalla'] for r in resultados])

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
    st.markdown('<div class="section-header">Parámetro calculado</div>', unsafe_allow_html=True)
    st.latex(r"d = \frac{\partial v}{\partial x} + \frac{\partial u}{\partial y}")
    st.caption("Diferencias finitas centradas · orden 2")
    st.divider()

    st.markdown('<div class="section-header">Controles del mapa</div>', unsafe_allow_html=True)
    show_vec   = st.checkbox("Vectores de viento", True)
    show_iso   = st.checkbox("Isolíneas de d", True)
    show_zero  = st.checkbox("Isolínea d = 0", True)
    show_brd   = st.checkbox("Bordes descartados", True)
    n_iso = st.slider("N° de isolíneas", 5, 22, 14)
    alpha = st.slider("Opacidad relleno", 0.3, 1.0, 0.75, 0.05)
    sc_v  = st.slider("Escala vectores", 0.04, 0.22, 0.10, 0.01)
    dpi   = st.select_slider("Resolución (DPI)", options=[80,100,120,150], value=100)
    st.divider()

    st.markdown('<div class="section-header">Estadísticas FL300</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    c1.metric("Mínimo d", f"{vals.min():.3f}")
    c2.metric("Máximo d", f"{vals.max():.3f}")
    c1.metric("Media d",  f"{vals.mean():.3f}")
    c2.metric("σ d",      f"{vals.std():.3f}")
    st.metric("Puntos calculados",   f"{len(resultados)}")
    st.metric("Bordes descartados",  f"{len(descartados)}")
    st.caption("Unidades: ×10⁻⁵ s⁻¹")

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "Mapa de cizalla",
    "Componentes u y v",
    "Ejemplo numérico",
    "Código esencial",
    "Tabla de datos",
])

# ─────────────────────────────────────────────────────────────────────────────
# Función genérica de mapa con Cartopy
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def make_map(lat_mn,lat_mx,lon_mn,lon_mx,n_iso_,alpha_,sc_,
             sv,si,sz,sb,dpi_,campo='cizalla',titulo='',label_cb='',cmap='RdBu_r',div=True):
    lats_i = np.array([r['lat']      for r in resultados])
    lons_i = np.array([r['lon']      for r in resultados])
    vals_i = np.array([r[campo]      for r in resultados])

    lon_g = np.linspace(lon_mn,lon_mx,200)
    lat_g = np.linspace(lat_mn,lat_mx,150)
    LON_G,LAT_G = np.meshgrid(lon_g,lat_g)
    FIELD = griddata((lons_i,lats_i),vals_i,(LON_G,LAT_G),method='linear')

    proj = ccrs.PlateCarree()
    fig  = plt.figure(figsize=(14,7.5),dpi=dpi_)
    ax   = fig.add_subplot(1,1,1,projection=proj)
    ax.set_extent([lon_mn,lon_mx,lat_mn,lat_mx],crs=proj)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'),facecolor='#d6eaf8',zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('50m'),facecolor='#f2ece4',zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.7,edgecolor='#444',zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),lw=0.35,edgecolor='#888',ls=':',zorder=5)
    ax.add_feature(cfeature.LAKES.with_scale('50m'),facecolor='#d6eaf8',lw=0.3,zorder=3)

    vmax = max(abs(float(np.nanmin(FIELD))),abs(float(np.nanmax(FIELD)))) if div \
           else float(np.nanmax(FIELD))
    norm = TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax) if div \
           else plt.Normalize(0,vmax)
    cm   = plt.get_cmap(cmap)

    cf = ax.contourf(LON_G,LAT_G,FIELD,levels=28,cmap=cm,norm=norm,
                     transform=proj,alpha=alpha_,zorder=2,extend='both')

    if si:
        cl=ax.contour(LON_G,LAT_G,FIELD,levels=n_iso_,
                      colors='#1a1a2e',linewidths=0.45,transform=proj,zorder=6,alpha=0.6)
        ax.clabel(cl,fmt='%.1f',fontsize=6,inline=True,inline_spacing=1)
    if sz and div:
        ax.contour(LON_G,LAT_G,FIELD,levels=[0],colors='#111',
                   linewidths=1.5,linestyles='--',transform=proj,zorder=7)

    for lat_r,lbl,col in [(0,'Ecuador','#c0392b'),(23.5,'T. Cáncer','#e74c3c'),(-23.5,'T. Capricornio','#2471a3')]:
        if lat_mn<=lat_r<=lat_mx:
            ax.plot([lon_mn,lon_mx],[lat_r,lat_r],color=col,lw=0.9,ls='--',transform=proj,alpha=0.75,zorder=4)
            ax.text(lon_mn+1,lat_r+0.8,lbl,fontsize=6.5,color=col,transform=proj)

    if sv:
        for r in resultados:
            if not (lat_mn<=r['lat']<=lat_mx and lon_mn<=r['lon']<=lon_mx): continue
            uu=r['u_ms']*sc_; vv=r['v_ms']*sc_
            if abs(uu)+abs(vv)<0.03: continue
            ax.annotate('',xy=(r['lon']+uu,r['lat']+vv),xytext=(r['lon'],r['lat']),
                        arrowprops=dict(arrowstyle='-|>',color='#0c447c',lw=0.65,mutation_scale=6),
                        transform=proj,zorder=8)

    ax.scatter(lons_i,lats_i,c=vals_i,cmap=cm,norm=norm,
               s=20,zorder=9,edgecolors='#0c447c',lw=0.3,transform=proj)

    if sb and campo=='cizalla':
        lats_b=np.array([d[0] for d in descartados])
        lons_b=np.array([d[1] for d in descartados])
        mask=(lats_b>=lat_mn)&(lats_b<=lat_mx)&(lons_b>=lon_mn)&(lons_b<=lon_mx)
        if mask.any():
            ax.scatter(lons_b[mask],lats_b[mask],marker='X',color='#e67e22',s=55,
                       zorder=9,edgecolors='#8b4500',lw=0.8,transform=proj,
                       label=f'Borde descartado ({mask.sum()} pts)')
            ax.legend(loc='lower left',fontsize=8,framealpha=0.92,
                      edgecolor='#c8dff0',facecolor='#f0f6fc')

    cb=plt.colorbar(cf,ax=ax,pad=0.01,shrink=0.78,aspect=24,extend='both' if div else 'max')
    cb.set_label(label_cb,fontsize=9,labelpad=8,color='#0c447c')
    cb.ax.tick_params(labelsize=8)

    gl=ax.gridlines(draw_labels=True,lw=0.45,color='#5a7a9a',alpha=0.4,ls='--')
    gl.top_labels=False; gl.right_labels=False
    gl.xlocator=mticker.FixedLocator(range(-130,-30,10))
    gl.ylocator=mticker.FixedLocator(range(-25,50,5))
    gl.xlabel_style={'size':7.5,'color':'#3a5570'}
    gl.ylabel_style={'size':7.5,'color':'#3a5570'}

    ax.set_title(titulo,fontsize=10,fontweight='bold',pad=8,color='#0c447c')
    fig.patch.set_facecolor('#f5f9fd')
    ax.set_facecolor('#d6eaf8')

    plt.tight_layout(rect=[0,0.03,1,1])
    buf=io.BytesIO()
    fig.savefig(buf,format='png',bbox_inches='tight',facecolor='#f5f9fd',dpi=dpi_)
    plt.close(fig); buf.seek(0)
    return buf

# ── TAB 1 — Mapa de cizalla ───────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="info-strip">Rojo = d &gt; 0 &nbsp;·&nbsp; Azul = d &lt; 0 &nbsp;·&nbsp; Línea discontinua = d = 0 &nbsp;·&nbsp; X naranja = borde descartado (60 pts)</div>', unsafe_allow_html=True)
    buf = make_map(
        lat_mn=-28, lat_mx=48, lon_mn=-132, lon_mx=-33,
        n_iso_=n_iso, alpha_=alpha, sc_=sc_v,
        sv=show_vec, si=show_iso, sz=show_zero, sb=show_brd, dpi_=dpi,
        campo='cizalla',
        titulo=r'$d = \partial v/\partial x + \partial u/\partial y$ '
               f'| Dif. finitas centradas | {len(resultados)} pts | {len(descartados)} bordes descartados',
        label_cb='Deformación por Cizalla (×10⁻⁵ s⁻¹)',
        cmap='RdBu_r', div=True
    )
    st.image(buf, use_container_width=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Mínimo",  f"{vals.min():.4f} ×10⁻⁵ s⁻¹")
    c2.metric("Máximo",  f"{vals.max():.4f} ×10⁻⁵ s⁻¹")
    c3.metric("Media",   f"{vals.mean():.4f} ×10⁻⁵ s⁻¹")
    c4.metric("σ",       f"{vals.std():.4f} ×10⁻⁵ s⁻¹")

    buf_dl = make_map(
        lat_mn=-28, lat_mx=48, lon_mn=-132, lon_mx=-33,
        n_iso_=14, alpha_=0.75, sc_=0.10,
        sv=True, si=True, sz=True, sb=True, dpi_=150,
        campo='cizalla', titulo='', label_cb='d (×10⁻⁵ s⁻¹)',
        cmap='RdBu_r', div=True
    )
    st.download_button("Descargar mapa PNG (150 DPI)", data=buf_dl,
                       file_name="mapa_cizalla_FL300.png", mime="image/png")

# ── TAB 2 — Componentes u y v ─────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="info-strip">u = −V·sin(θ) &nbsp;·&nbsp; v = −V·cos(θ) &nbsp;·&nbsp; Signo negativo: la dirección meteorológica indica de dónde viene el viento</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Componente zonal u (m/s)</div>', unsafe_allow_html=True)
        bu = make_map(
            lat_mn=-28, lat_mx=48, lon_mn=-132, lon_mx=-33,
            n_iso_=10, alpha_=0.78, sc_=0.09,
            sv=False, si=False, sz=True, sb=False, dpi_=100,
            campo='u_ms', titulo='u = −V·sin(θ) | Componente zonal (m/s)',
            label_cb='u (m/s)', cmap='RdBu_r', div=True
        )
        st.image(bu, use_container_width=True)
        st.caption("Azul = flujo al Este · Rojo = flujo al Oeste")
    with c2:
        st.markdown('<div class="section-header">Componente meridional v (m/s)</div>', unsafe_allow_html=True)
        bv = make_map(
            lat_mn=-28, lat_mx=48, lon_mn=-132, lon_mx=-33,
            n_iso_=10, alpha_=0.78, sc_=0.09,
            sv=False, si=False, sz=True, sb=False, dpi_=100,
            campo='v_ms', titulo='v = −V·cos(θ) | Componente meridional (m/s)',
            label_cb='v (m/s)', cmap='BrBG', div=True
        )
        st.image(bv, use_container_width=True)
        st.caption("Azul = flujo al Norte · Rojo = flujo al Sur")

    st.markdown('<div class="section-header">Velocidad escalar |V| (m/s)</div>', unsafe_allow_html=True)
    bs = make_map(
        lat_mn=-28, lat_mx=48, lon_mn=-132, lon_mx=-33,
        n_iso_=10, alpha_=0.78, sc_=0.09,
        sv=False, si=False, sz=False, sb=False, dpi_=100,
        campo='spd_ms', titulo='|V| = √(u²+v²) | Velocidad escalar (m/s)',
        label_cb='|V| (m/s)', cmap='YlOrRd', div=False
    )
    st.image(bs, use_container_width=True)
    st.caption("Amarillo intenso = chorro subtropical (40°N/105°W) · Azul = zona tropical de baja velocidad")

# ── TAB 3 — Ejemplo numérico ──────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="info-strip">Nodo de máxima cizalla del dominio — 40°N / 105°W &nbsp;·&nbsp; Código F300: <b>31064M46</b></div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
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
        u = −32.92 × sin(310°) = −32.92 × (−0.766) = <span class="val">+25.22 m/s</span><br>
        v = −32.92 × cos(310°) = −32.92 × (+0.643) = <span class="val">−21.16 m/s</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Paso 4 — Los cuatro vecinos necesarios</div>', unsafe_allow_html=True)
    import pandas as pd
    df_v = pd.DataFrame([
        {"Vecino":"Este  40°N/100°W","Código":"20042M47","Dir/Vel":"200°/42 kt","u (m/s)":"—","v (m/s)":"+20.30"},
        {"Vecino":"Oeste 40°N/110°W","Código":"29060M46","Dir/Vel":"290°/60 kt","u (m/s)":"—","v (m/s)":"−10.56"},
        {"Vecino":"Norte 45°N/105°W","Código":"29069M47","Dir/Vel":"290°/69 kt","u (m/s)":"+33.36","v (m/s)":"—"},
        {"Vecino":"Sur   35°N/105°W","Código":"30062M44","Dir/Vel":"300°/62 kt","u (m/s)":"+27.62","v (m/s)":"—"},
    ])
    st.dataframe(df_v, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="step-card">
      <div class="step-num">Paso 5 — Distancias (1° = 110 km)</div>
      <div class="step-result">
        2Δy = 2 × 5 × 110 000 = <span class="val">1 100 000 m</span><br>
        2Δx = 1 100 000 × cos(40°) = <span class="val">842 649 m</span>
      </div>
    </div>
    <div class="step-card">
      <div class="step-num">Paso 6 — Diferencias finitas centradas</div>
      <div class="step-result">
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

# ── TAB 4 — Código esencial ───────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="info-strip">Solo las tres partes fundamentales del método — independientes de la lectura del archivo y de la visualización</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Parte 1 — Conversión a componentes u, v</div>', unsafe_allow_html=True)
    st.code("""# Dirección indica DESDE donde viene → signo negativo
KT_TO_MS = 0.514444   # 1 kt = 0.514444 m/s

def wind_to_uv(direction_deg, speed_kt):
    spd_ms = speed_kt * KT_TO_MS
    theta  = math.radians(direction_deg)
    u = -spd_ms * math.sin(theta)   # zonal   (Este+)
    v = -spd_ms * math.cos(theta)   # meridional (Norte+)
    return u, v""", language='python')

    st.markdown('<div class="section-header">Parte 2 — Distancias en metros</div>', unsafe_allow_html=True)
    st.code("""DEG_M = 110_000.0   # 1° = 110 km (indicación del profesor)
DLAT  = 5.0          # paso de malla latitudinal (°)
DLON  = 5.0          # paso de malla longitudinal (°)

def dos_dx(lat):
    # Varía con cos(lat) → esfera terrestre
    return 2 * DLON * DEG_M * math.cos(math.radians(lat))

def dos_dy():
    return 2 * DLAT * DEG_M   # = 1 100 000 m (constante)""", language='python')

    st.markdown('<div class="section-header">Parte 3 — Diferencias finitas centradas</div>', unsafe_allow_html=True)
    st.code("""for (lat, lon), pt in grid.items():
    v_este  = grid.get((lat, lon+5), {}).get('v')   # vecino derecho
    v_oeste = grid.get((lat, lon-5), {}).get('v')   # vecino izquierdo
    u_norte = grid.get((lat+5, lon), {}).get('u')   # vecino arriba
    u_sur   = grid.get((lat-5, lon), {}).get('u')   # vecino abajo

    # Condición estricta: los 4 vecinos deben existir
    if any(x is None for x in [v_este, v_oeste, u_norte, u_sur]):
        continue   # borde → descartado (60 puntos)

    dvdx = (v_este  - v_oeste) / dos_dx(lat)   # ∂v/∂x
    dudy = (u_norte - u_sur)   / dos_dy()      # ∂u/∂y
    d    = (dvdx + dudy) * 1e5                 # ×10⁻⁵ s⁻¹""", language='python')

# ── TAB 5 — Tabla de datos ────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Filtros</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    lat_f = c1.slider("Latitud", -25, 45, (-25,45))
    spd_f = c2.slider("Velocidad (kt)", 0, 130, (0,130))
    ciz_f = c3.slider("Cizalla (×10⁻⁵ s⁻¹)", -4.0, 4.5, (-4.0,4.5), 0.1)

    df = pd.DataFrame(resultados)
    df.columns = ['Lat°N','Lon°W','Dir°','kt','m/s','u (m/s)','v (m/s)','d (×10⁻⁵ s⁻¹)']
    df['Lon°W'] = df['Lon°W'].abs()
    mask = (
        (df['Lat°N']>=lat_f[0])&(df['Lat°N']<=lat_f[1])&
        (df['kt']>=spd_f[0])&(df['kt']<=spd_f[1])&
        (df['d (×10⁻⁵ s⁻¹)']>=ciz_f[0])&(df['d (×10⁻⁵ s⁻¹)']<=ciz_f[1])
    )
    df_f = df[mask]
    st.caption(f"{len(df_f)} de {len(df)} puntos")

    st.dataframe(
        df_f.style.background_gradient(subset=['d (×10⁻⁵ s⁻¹)'],cmap='RdBu_r'),
        use_container_width=True, height=380
    )
    csv = df_f.to_csv(index=False).encode('utf-8')
    st.download_button("Descargar CSV",data=csv,
                       file_name="fl300_cizalla_g5.csv",mime="text/csv")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Grupo 5 — Meteorología Aeronáutica 2026 · WINTEM_2026-1_FL300_G5__1_.txt · d = ∂v/∂x + ∂u/∂y · 1 kt = 0.5144 m/s · 1° = 110 km")