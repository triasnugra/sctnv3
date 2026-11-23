# CryptoEliz Scalp Engine v2 (sctnv2)

Mesin **analisis** trading scalping crypto (BTC + alts terpilih) multi-timeframe, berbasis aturan yang
dioperasionalkan dari filosofi trader @eliz883 (CryptoEliz). Output: laporan Markdown, chart beranotasi,
ringkasan signal (**LONG / SHORT / NO TRADE**) beserta alasan dan rincian skor — siap dijalankan lewat GitHub Actions.

**v2 = sctn + indikator pada grafik** (MACD, OBV, RSI, EMA 12/26/50/100/200, Supertrend, UCTS≈, Spaceman BTC Key Levels, Volume + Volume Profile). Logika signal, skor, risk, dan forward test **tidak berubah** — lihat bagian 2.

> ⚠️ **Bukan jaminan profit.** Semua angka (jumlah candle, % deviasi, ambang volume, dst.) adalah **hipotesis operasional**
> dari aturan yang aslinya diskresioner. Belum ada bukti edge. Gunakan untuk paper-trading / forward-test dulu.
> Bukan nasihat finansial.

## 1. Filosofi → implementasi

| Prinsip | Implementasi di mesin |
|---|---|
| Trade di tepi range, bukan tengah | `range_detector` membuat box + edge zone 20%; entry di tengah range = **veto** (kecuali breakout-retest) |
| Trigger dulu, bukan prediksi | Hanya 3 trigger: deviation+reclaim, breakout+retest, fib 0.75 **dengan reaksi candle**. Limit fib tanpa reaksi = hanya konteks |
| Invalidasi sebelum target | Stop = titik invalidasi struktural (+buffer). Stop > 6% → NO TRADE. RR gabungan < 1.5 → NO TRADE |
| Bias sederhana | `multi_tf_bias` (Weekly+Daily): di atas key level → cari long, di bawah → short. Trade melawan bias = veto |
| Patience is a position | Default **NO TRADE**. Perlu bias + trigger + skor ≥ ambang + lolos semua veto |
| Pisahkan investasi vs trading | Mesin ini hanya trading/scalp. Spot akumulasi di demand zone di luar cakupan |

**Bias (HTF) dan trigger (LTF) dipisah tegas:** Weekly/Daily hanya memberi arah; 4H memberi struktur/range; 1H dan 15m memberi trigger.

**Tidak pernah jadi trigger:** liquidity heatmap, opini makro/konspirasi/komentar non-teknikal. (Tidak diimplementasikan sama sekali.)

## 2. Indikator pada grafik (baru di v2)

Sumber daftar indikator: tweet @eliz883 (Apr 2022 & Mei 2023). Ia sendiri menyebut indikator hanya mengerjakan ~5% — *95% price action* —
karena itu di mesin ini indikator adalah **lapisan tampilan**: tidak masuk skor, veto, risk plan, maupun `config_hash`.

| Indikator | Tampil di | Default |
|---|---|---|
| EMA 12 / 26 / 50 / 100 / 200 | overlay panel harga (EMA200 tebal gelap) | semua aktif; dihitung di seluruh histori agar "hangat", EMA yang datanya kurang tidak digambar |
| Supertrend | overlay panel harga, hijau = uptrend / merah = downtrend | ATR 10 × 3.0 (default TradingView) |
| Volume Profile | histogram abu di kiri + garis POC / VAH / VAL | 48 bin, value area 70%, range = candle yang tampil |
| Spaceman BTC Key Levels | garis + label di margin kanan | Day/Week/Month Open, High/Low/Mid periode sebelumnya, Monday H/L/Mid, Year Open |
| UCTS≈ | marker ▲ Buy / ▼ Sell | **proksi** (lihat catatan) |
| Volume | panel bawah (bar + SMA 20) | aktif |
| OBV | panel bawah | aktif |
| RSI | panel bawah | 14, batas 70/30 |
| MACD | panel bawah (histogram + garis) | 12/26/9 |

Dashboard menampilkan chart **15m / 1h / 4h** per simbol (tombol pilih TF), tabel pembacaan indikator per TF, strip Supertrend multi-TF,
daftar Key Levels beserta jarak ke harga, dan ringkasan Volume Profile. `report.md` memuat tabel yang sama.
Semua diatur di `config/chart_indicators.yaml` (nyalakan/matikan tiap indikator, ubah periode, pilih panel & TF).

**Catatan jujur tentang dua indikator:**
- **UCTS** (Ultimate Crypto Trading Strategy, CryptoNTez) adalah skrip TradingView *invite-only* — kodenya tertutup, jadi **tidak bisa direplikasi persis**.
  Yang ada di sini adalah **proksi terbuka** bertanda `UCTS≈`: *Buy* = Supertrend berbalik naik + histogram MACD sedang naik; *Sell* = kebalikannya.
  Jangan samakan dengan sinyal UCTS asli. (Di tweet tertulis "UTCS"; diasumsikan itu UCTS.)
- **Spaceman BTC Key Levels** adalah indikator open-source "Key Levels SpacemanBTC IDWM". Yang direplikasi adalah *konsepnya*
  (level Open/High/Low/Mid per Day/Week/Month + Monday), bukan salinan kodenya. Waktu memakai UTC; level yang datanya tidak mencakup
  satu periode penuh dilewati, bukan ditebak.

**Jaminan "tampilan saja" (diuji otomatis di `tests/test_indicators.py`):** kode di `src/analysis`, `src/signal`, `src/evaluate`, `src/backtest`
tidak boleh mengimpor lapisan indikator; `chart_indicators.yaml` tidak masuk `config_hash`; signal identik dengan indikator ON/OFF.
Hash config sctnv2 sama dengan sctn selama `parameters.yaml`/`symbols.yaml`/`risk.yaml` tidak diubah.

## 2b. Scanner universe otomatis (baru di v2)

Koin yang dianalisis dipilih otomatis (default: **top 30 volume 24 jam di Binance**, cadangan Bitget). Atur di blok `scanner:` pada `config/symbols.yaml`.

| `sources` | Isi | Butuh |
|---|---|---|
| `binance_volume` | top N pasangan USDT berdasarkan quoteVolume 24 jam | — |
| `bitget_volume` | top N pasangan USDT berdasarkan usdtVolume 24 jam | — |
| `coinmarketcap_rank` | peringkat CMC 1..N (market cap), disaring yang punya pasangan USDT di bursa | secret `CMC_API_KEY` (key gratis di pro.coinmarketcap.com; 1 panggilan/hari jauh di bawah kuota gratis) |

`sources` dicoba berurutan sampai ada yang berhasil. Contoh peringkat CMC: `sources: ["coinmarketcap_rank", "binance_volume"]`.
Untuk CMC: Settings → Secrets and variables → Actions → New repository secret → `CMC_API_KEY`.

**Aturan yang menjaga validitas forward test**
- **Refresh sekali per 24 jam** (`refresh_hours`), disimpan di `output/universe.json` (di-commit workflow) → daftar stabil sepanjang hari, bisa direproduksi.
  Setiap refresh dicatat di `output/universe_history.csv` (waktu, sumber, daftar, koin masuk/keluar) sebagai jejak audit.
- Peringkat hanya memakai data saat refresh → tidak ada lookahead terhadap sinyal yang dicatat sesudahnya.
- **`config_hash` memuat ATURAN scanner** (sumber, top_n, filter), bukan daftar koinnya. Koin keluar/masuk tidak mereset forward test; mengubah aturan (mis. top_n 30→20) mereset. Hash sctnv2 kini berbeda dari sctn — memang seharusnya, karena universe-nya berbeda.
- Dikeluarkan otomatis: stablecoin, token wrapped/staked, token emas, token leveraged, pasangan non-USDT, dan koin dengan volume 24j < `min_quote_volume_usdt` (5 juta USDT).
- **Bila sumber gagal**: sumber berikutnya → cache terakhir (aturan sama) → daftar statis `symbols`. Fallback selalu ditandai (`status`) di `signals.json`, laporan, dan banner dashboard — tidak pernah diam-diam.
- **Backtest tetap memakai daftar statis `symbols`**: top-30 pada tanggal lampau tidak diketahui, memakai daftar hari ini akan menimbulkan survivorship bias.
- Override manual (input `symbols` di workflow / `--symbols`) mematikan scanner untuk run itu. `enabled: false` mengembalikan perilaku daftar statis, dengan `config_hash` identik dengan sctn.

**Catatan membandingkan beberapa mesin (sctn / sctnv2 / sctnv4)**
- Universe berbeda = sampel berbeda. Bandingkan pada **simbol yang sama** (mis. irisan dengan 8 koin lama), atau samakan universe antar mesin. Opsi `always_include` memaksa koin tertentu selalu ikut.
- Sinyal antar altcoin berkorelasi kuat dengan BTC, jadi 30 koin bukan 30 sampel independen: interval kepercayaan dari jumlah trade bisa terlalu optimis.
- Memilih "yang paling valid" dari 3 mesin adalah perbandingan ganda; tetapkan kriteria lolos (gate yang sudah ada) **sebelum** melihat hasil.

**Ongkos:** ±1 menit untuk 30 koin pada data sintetis (data live lebih lama karena unduhan candle); artefak situs ±30 MB per deploy (90 chart). Bila terlalu berat, set `extra_timeframes: []` di `config/chart_indicators.yaml` atau turunkan `top_n`.

## 3. Struktur repo

```
index.html                              # DASHBOARD (GitHub Pages) — statis, tanpa dependensi
.github/workflows/run-analysis.yml      # analisis terjadwal + forward test + build & deploy dashboard
.github/workflows/run-backtest.yml      # backtest walk-forward (manual)
config/{symbols,parameters,risk}.yaml   # SEMUA parameter analisis (tidak ada hard-code) — masuk config_hash
config/chart_indicators.yaml            # v2: indikator chart (tampilan saja) — TIDAK masuk config_hash
                                        # symbols.yaml berisi blok `scanner:` (universe otomatis) — aturannya masuk config_hash
src/data/        fetch_ohlcv.py (Binance REST + ccxt fallback, paging), indicators.py (+RSI, MACD, Supertrend, Volume Profile),
                 key_levels.py (Spaceman Key Levels), universe.py (scanner universe), synthetic.py (demo)
src/analysis/    range_detector, liquidity_deviation, breakout_retest, fibonacci, monday_range,
                 three_tap_wolf, fvg, obv_confirmation, multi_tf_bias, pipeline
src/signal/      signal_engine.py (skor + veto), risk_manager.py (stop/target/size/filter/fee)
src/evaluate/    simulator.py (hasil trade candle-by-candle), stats.py (R, PF, SQN, CI95), forward_test.py
src/backtest/    run.py (replay as-of tanpa lookahead)
src/report/      generate_report.py, chart_generator.py (chart multi-panel), indicator_suite.py (hitung & ringkas indikator)
src/notify.py    Telegram / Discord (opsional)
src/site_builder.py  # rakit folder site/ untuk Pages
tests/           60 unit + smoke test (termasuk invarian 'indikator = tampilan saja')
output/          signal_log.csv (jurnal), forward_test.json, backtest/ (di-commit); latest/ (tidak di-commit)
```

## 4. Definisi operasional (hipotesis — semua bisa di-tune di `config/parameters.yaml`)

| Konsep | Definisi default |
|---|---|
| **Range** | Window 48–168 candle H1 (4H: 30–90). ≥2 pivot menyentuh tiap tepi (toleransi 0.35%), lebar 1.5–12%, drift < 0.6 × tinggi box, ≤3% close di luar box. `exclude_recent` candle terbaru tidak ikut membentuk box (agar deviasi/breakout terbaca) |
| **Deviation + reclaim** | Wick tembus level ≥0.15% (maks 3%), candle sebelumnya masih di dalam, close kembali di dalam ≤4 candle (H1). Entry = retest level ±0.20%; stop = di luar ekstrem deviasi; invalid jika ada close melewati ekstrem |
| **Breakout + retest** | Close lewat level ≥0.30%, body ≥0.6 ATR, volume ≥1.2× SMA20, bertahan 2 candle (kalau tidak → itu deviation). Entry di retest ±0.20% |
| **Fib 0.75** | Impulse ≥5% dari pivot; zona 0.71–0.79; invalid di 0.90. Trigger hanya jika zona tersentuh + candle terakhir searah |
| **Monday range** | High/Low candle harian Senin (UTC) yang sudah close; fallback ke Senin minggu lalu |
| **Three-tap** | 3 pivot berurutan naik/turun (langkah 0.15–3%); tap-3 gagal: close kembali melewati tap-2 |
| **Wolf** | 5 pivot selang-seling, garis 1-3 & 2-4 konvergen, titik 5 overshoot garis 1-3 lalu bertahan |
| **FVG** | Gap 3 candle ≥0.25 ATR, belum terisi penuh; hanya konfirmasi |
| **OBV** | Divergensi pivot-ke-pivot; pola V: ayunan ≥3× rata-rata volume, pulih ≥70% |
| **Volume rendah** | Rata-rata 3 candle H1 terakhir < 0.5× baseline 50 candle → blok entry |

## 5. Scoring (transparan)

Setiap setup (arah × trigger) dinilai sendiri; item & poin ditampilkan di laporan.

`bias searah +2` · `trigger: deviation +3 / breakout-retest +2 / fib +2` · `harga di entry zone +1` ·
`konfluensi (tepi range 4H, Monday level, fib zone) +1 each` · `konfirmasi (FVG, OBV div, OBV-V, three-tap, wolf, volume spike) +1 each` ·
`konflik dengan bias BTC −1` (untuk alts).

Lolos jika **skor ≥ 6**, ada ≥1 konfluensi/konfirmasi tambahan, dan tidak ada veto:
bias berlawanan · entry di tengah range · status `late`/`invalidated`/`deep_retest` · level pending > 3 ATR ·
stop > 6% · RR gabungan < 1.5 · volume rendah · menjelang/sesudah monthly close · event blackout manual.

## 6. Manajemen posisi (risk.yaml)

Stop struktural + buffer (maks 6%); TP1 di level struktur terdekat dengan R ≥ 1.2 (fallback 1.5R), tutup 60%;
moonbag ke TP2 dengan stop dipindah ke BE+0.1%; leverage disarankan sedemikian hingga jarak likuidasi ≥ 2× jarak stop
(maks 10x). Lebih baik miss daripada stop lebar/likuidasi.

**Event besar (FOMC dll):** isi manual di `risk.yaml → filters.event_blackouts` (UTC). Tidak ada tanggal yang di-hardcode.

## 7. Setup GitHub (urutan)

1. Upload seluruh isi folder ini ke **root** repo (`index.html`, `config/`, `src/`, `.github/` harus langsung di root, bukan di dalam
   subfolder `sctnv2/`). Folder `.github` bersifat tersembunyi dan sering terlewat saat drag-and-drop di web — cara paling aman lewat git:
   ```bash
   unzip sctnv2.zip && cd sctnv2
   git init -b main && git add -A && git commit -m "init"
   git remote add origin https://github.com/triasnugra/sctnv2.git
   git push -u origin main --force
   ```
   Pastikan di GitHub tab **Actions** terlihat workflow `run-analysis` dan `run-backtest`.
2. **Settings → Actions → General → Workflow permissions → Read and write permissions.**
3. **Settings → Pages → Build and deployment → Source: GitHub Actions** (WAJIB — tanpa ini job `deploy` gagal dan situs tidak terbit).
4. Push pertama otomatis menjalankan `run-analysis`. Bisa juga manual: **Actions → run-analysis → Run workflow** (centang `demo` untuk uji data sintetis).
5. Dashboard: `https://triasnugra.github.io/sctnv2/` (muncul setelah job **deploy** hijau; lihat link di ringkasan run).
6. (Opsional) Notifikasi: *Settings → Secrets and variables → Actions → New repository secret*:
   `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`, dan/atau `DISCORD_WEBHOOK_URL`. Hanya signal **baru** yang dikirim.
7. (Disarankan) **Actions → run-backtest** (default 45 hari). Setelah selesai, workflow ini otomatis memicu `run-analysis`
   supaya dashboard memuat hasil backtest (matikan lewat input `publish`).

Jadwal default tiap jam (menit ke-5). Ubah di `run-analysis.yml` (`5 */4 * * *` = tiap 4 jam).
Job `deploy` hanya berjalan dari branch default (`main`) karena aturan environment `github-pages`.

### Migrasi dari sctn
sctnv2 adalah repo baru dengan **jurnal & forward test kosong** (folder `output/` sengaja tidak membawa data sctn). Karena `config_hash` sama,
bila ingin melanjutkan jurnal lama, salin `output/signal_log.csv` dan `output/forward_test.json` dari repo sctn sebelum push pertama.
Bila tidak, forward test sctnv2 mulai dari nol (sampel 150 trade / 20 minggu dihitung ulang).

### Apa yang di-commit ke repo
Hanya state kecil: `output/signal_log.csv`, `output/forward_test.json`, `output/backtest/*`. Chart PNG & laporan
**tidak** di-commit (dikirim ke dashboard lewat Pages artifact) supaya riwayat git tidak membengkak.

### Lokal
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests
python -m src.main --demo                        # data SINTETIS (tanpa internet); chart 15m/1h/4h per simbol di output/latest/charts/
python -m src.main                               # live
python -m src.evaluate.forward_test              # evaluasi jurnal
python -m src.backtest.run --days 45             # backtest (tambah --demo untuk uji)
python -m src.site_builder && python -m http.server -d site 8000   # dashboard di http://localhost:8000
```
Dashboard TIDAK bisa dibuka lewat `file://` (browser memblokir fetch); gunakan server lokal di atas.

### Sumber data
`binance_rest` mencoba beberapa host Binance (termasuk `data-api.binance.vision`), lalu fallback `ccxt:okx`, `ccxt:kucoin`
(ubah di `symbols.yaml → data.providers`). Binance sering memblokir IP US (HTTP 451). Satu simbol selalu diambil
dari **satu** provider untuk semua TF. Hanya candle yang sudah close yang dianalisis.

## 8. Evaluasi: forward test & backtest

- **Jurnal** (`signal_log.csv`): hanya signal LONG/SHORT yang *baru pertama kali* muncul (dedup per kejadian setup lewat `signal_id`). Mode demo tidak menulis jurnal.
- **Forward test** (`forward_test.json`): tiap signal disimulasikan dengan candle 15m aktual. Aturan konservatif:
  stop & target di candle yang sama → stop dulu; limit pending batal bila TP1 tercapai sebelum terisi / lewat 24 jam;
  TP1 menutup 60% lalu stop ke BE+; timeout 72 jam = mark-to-market; fee+slippage kasar 0.10% dikonversi ke R.
- **Statistik**: expectancy (R), win rate, profit factor, SQN, CI95 expectancy, max drawdown (R).
  Status **UNVERIFIABLE** sampai ≥150 trade & ≥20 minggu; baru setelah itu PASS/FAIL (CI95 bawah > 0 dan PF ≥ 1.2).
- **Backtest**: replay per jam; pada tiap titik mesin hanya melihat candle yang sudah close saat itu (tanpa lookahead),
  memakai kode analisis & simulator yang sama dengan live. Hasilnya bukan bukti edge — sampel kecil, signal berkorelasi,
  parameter adalah hipotesis. Jangan tune parameter berdasarkan satu backtest (data snooping); catat `config_hash`.
- **Filter fee**: signal dengan stop terlalu rapat (fee > 0.15R) di-veto — pada stop sempit fee bisa memakan seluruh edge.

## 9. Keterbatasan
- Aturan diskresioner tidak sepenuhnya bisa direpresentasikan; hasil bisa berbeda dari penilaian trader aslinya.
- Simulasi tidak memodelkan funding, slippage nyata, partial fill, atau gap likuiditas.
- Jadwal cron GitHub bisa tertunda; signal H1/15m bisa basi — selalu cek chart & harga terkini sebelum bertindak.
- Estimasi likuidasi kasar; fee/funding/slippage tidak dimodelkan.
