import streamlit as st
import pandas as pd
from db_manager import DBManager
import time
from datetime import datetime, timedelta

# Sayfa ayarları
st.set_page_config(
    page_title="Crypto News Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=300) # 5 dakika cache
def load_data(hours):
    """Loads and processes data from the database."""
    with DBManager() as db:
        return db.get_grouped_news_for_app(hours=hours)

# Ana başlık
st.title("📈 Crypto News Verification Dashboard")
st.markdown("AI-Grouped & Verified News from Multiple Sources")

# Sidebar controls
st.sidebar.title("⚙️ Controls & Filters")

# Timeframe filter
hours_filter = st.sidebar.slider(
    "Timeframe (Hours)", 
    min_value=1, max_value=168, value=24,
    help="Filter news published in the last X hours."
)

# Minimum sources filter
min_sources_filter = st.sidebar.slider(
    "Minimum Sources", 
    min_value=1, max_value=10, value=1,
    help="Only show news groups shared by at least this many sources."
)

# Sorting option
sort_by = st.sidebar.selectbox(
    "Sort By",
    ["Importance", "Recency", "Sources"],
    help="Change the order of the news feed."
)

# Manual refresh button
if st.sidebar.button("🔄 Refresh Now"):
    st.cache_data.clear()
    st.rerun()

auto_refresh = st.sidebar.checkbox("Auto-Refresh (60s)")

grouped_news = load_data(hours_filter)

if not grouped_news:
    st.warning("No news groups found for this time range.")
    st.info("Make sure the backend scheduler (`scheduler.py`) is running.")
else:
    # DataFrame oluştur ve kaynak sayısını hesapla
    df = pd.DataFrame(grouped_news)
    df['source_count'] = df['sources'].apply(len)
    
    # Minimum kaynak sayısına göre filtrele
    df_filtered = df[df['source_count'] >= min_sources_filter].copy()

    # Önem seviyesi skorları
    importance_order = {"Important": 3, "Medium": 2, "Unimportant": 1}
    df_filtered['importance_score'] = df_filtered['importance_label'].map(importance_order).fillna(0)
    
    def parse_tweet_time(time_str):
        try:
            if isinstance(time_str, str):
                return pd.to_datetime(time_str.replace('Z', '+00:00'), utc=True)
            return pd.to_datetime(time_str, utc=True)
        except:
            return pd.Timestamp.now(tz='UTC')
    
    # Zaman verisini işle
    df_filtered['tweet_datetime'] = df_filtered['tweet_time'].apply(parse_tweet_time)

    # Seçilen kritere göre sırala
    if sort_by == "Importance":
        df_sorted = df_filtered.sort_values(
            by=['importance_score', 'tweet_datetime', 'source_count'], 
            ascending=[False, False, False]
        )
    elif sort_by == "Recency":
        df_sorted = df_filtered.sort_values(by='tweet_datetime', ascending=False)
    else:
        df_sorted = df_filtered.sort_values(
            by=['source_count', 'importance_score', 'tweet_datetime'], 
            ascending=[False, False, False]
        )
    # Dashboard metrikleri
    st.subheader("Dashboard Metrics")
    col1, col2, col3 = st.columns(3)
    total_groups = len(df_sorted)
    multi_source_groups = len(df_sorted[df_sorted['source_count'] > 1])
    avg_confidence = df_sorted['importance_confidence'].mean() if not df_sorted.empty else 0
    total_authors = df_sorted['sources'].explode().nunique() if not df_sorted.empty else 0

    col1.metric("📰 News Groups", f"{total_groups}")
    col2.metric("🔗 Multi-Source Groups", f"{multi_source_groups}")
    col3.metric("🗞️ Total Sources", f"{total_authors}")
    st.divider()
    # Ana haber listesi
    st.subheader(f"Top News Stories from the Last {hours_filter} Hours")

    if df_sorted.empty:
        st.info("No news groups match your current filter settings.")
    else:
        # Her haber grubu için kart oluştur
        for _, group in df_sorted.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                # Sol kolon - Ana içerik
                with col1:
                    importance = group.get('importance_label', 'Unknown')
                    # Önem seviyesine göre badge
                    if importance == 'Important': badge = "🔥"
                    elif importance == 'Medium': badge = "📊"
                    else: badge = "📝"

                    display_title = group.get('llm_summary') or group.get('title', 'No Title')
                    st.markdown(f"##### {badge} {display_title}")

                    source_count = len(group['sources'])
                    if source_count > 1:
                        st.success(f"✅ **Verified by {source_count} different sources**")
                    else:
                        st.info(f" Shared by 1 source")
                    
                    st.write("") 
                    st.caption("Original Text Snippet:")
                    # İçerik önizlemesi (280 karakter limit)
                    content = group.get('content', '')
                    st.markdown(f"> {content[:280]}..." if len(content) > 280 else f"> {content}")
                # Sağ kolon - Meta bilgiler
                with col2:
                    st.markdown(f"**Importance Level**")
                    st.markdown(f"### {importance}")
                    st.link_button("🔗 Go to Main Story", group['url'], use_container_width=True)
                    # Yayın zamanı
                    try:
                        dt_object = datetime.fromisoformat(group['tweet_time'].replace('Z', '+00:00'))
                        turkey_time = dt_object + timedelta(hours=3)
                        st.caption(f"Published: {turkey_time.strftime('%b %d, %H:%M')}")
                    except:
                        st.caption(f"Time: {group['tweet_time']}")
                    # Kaynak linkleri
                    st.markdown("**Sources:**")
                    # URL'ler varsa direkt link, yoksa Twitter profil linki
                    if 'source_urls' in group and group['source_urls']:
                        source_links = []
                        for source in group['sources']:
                            source_clean = source.strip()
                            if source_clean in group['source_urls']:
                                tweet_url = group['source_urls'][source_clean]
                                source_links.append(f"[{source_clean}]({tweet_url})")
                            else:
                                profile_clean = source_clean.lstrip('@')
                                source_links.append(f"[{source_clean}](https://twitter.com/{profile_clean})")
                        st.markdown(", ".join(source_links))
                    else:
                        source_links = []
                        for source in group['sources']:
                            source_clean = source.strip().lstrip('@')
                            source_links.append(f"[{source.strip()}](https://twitter.com/{source_clean})")
                        st.markdown(", ".join(source_links))

            st.write("")

# Otomatik yenileme (eğer aktifse)
if auto_refresh:
    time.sleep(60)
    st.rerun()
