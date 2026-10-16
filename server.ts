import express, { Request, Response } from 'express';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

app.use(express.json({ limit: '25mb' }));

// Initialize Google GenAI client
const apiKey = process.env.GEMINI_API_KEY || '';
const ai = new GoogleGenAI({
  apiKey,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

// Helper to check if API key is configured
const hasApiKey = () => Boolean(process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== 'MY_GEMINI_API_KEY');

// API Endpoint 1: Text-to-Speech using gemini-3.8-flash-tts
app.post('/api/tts', async (req: Request, res: Response) => {
  try {
    const { text, speaker = 'Alex' } = req.body;
    if (!text || typeof text !== 'string') {
      res.status(400).json({ error: 'Parameter text diperlukan' });
      return;
    }

    if (!hasApiKey()) {
      res.status(503).json({
        fallback: true,
        message: 'Kunci API Gemini belum diatur di Secrets. Menggunakan browser Web Speech API bawaan.',
      });
      return;
    }

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash-tts',
      contents: [
        {
          role: 'user',
          parts: [
            {
              text: `Bacakan teks edukasi anak SD berikut ini dengan intonasi ramah, gembira, jelas, dan mendidik seperti guru IPAS yang menyenangkan: "${text.slice(0, 300)}"`,
              speechMetadata: {
                speaker: speaker,
                style: 'Cheerful, warm, encouraging elementary school teacher for kids in Indonesian',
              },
            },
          ],
        },
      ],
      config: {
        responseModalities: ['AUDIO'],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: 'Kore' },
          },
        },
      },
    });

    const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (base64Audio) {
      res.json({
        success: true,
        audioBase64: base64Audio,
        mimeType: 'audio/wav',
      });
    } else {
      res.status(500).json({ fallback: true, error: 'Gagal membuat audio Gemini TTS' });
    }
  } catch (error: any) {
    console.error('Error in /api/tts:', error?.message || error);
    res.status(500).json({ fallback: true, error: error?.message || 'Gagal memproses TTS' });
  }
});

// API Endpoint 2: Search Grounding for current climate facts using gemini-3.8-flash
app.post('/api/search-facts', async (req: Request, res: Response) => {
  try {
    const { query } = req.body;
    const userQuery = query || 'Fakta terkini perubahan iklim bumi dan suhu global untuk anak sekolah';

    if (!hasApiKey()) {
      res.json({
        success: true,
        text: 'Tahukah kamu? Suhu rata-rata bumi telah meningkat sekitar 1,1°C sejak abad ke-19 akibat gas rumah kaca berlebih. Menanam satu pohon dapat menyerap hingga 22 kg karbon dioksida setiap tahunnya!',
        sources: [
          { title: 'Kurikulum Merdeka IPAS Kelas 4 Bab 4', uri: 'https://kemdikbud.go.id' }
        ],
      });
      return;
    }

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: `Kamu adalah asisten guru cerdas IPAS Sekolah Dasar kelas 4. Jawablah pertanyaan anak tentang iklim berikut dengan bahasa yang sangat mudah dimengerti, ceria, singkat (maksimal 3 paragraf pendek), dan sertakan fakta nyata dari pencarian web: "${userQuery}". Hindari istilah rumit tanpa penjelasan.`,
      config: {
        tools: [{ googleSearch: {} }],
      },
    });

    const text = response.text || 'Tidak ada informasi ditemukan.';
    const rawChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
    const sources = rawChunks
      .filter((chunk: any) => chunk.web?.uri)
      .map((chunk: any) => ({
        title: chunk.web.title || 'Sumber Informasi',
        uri: chunk.web.uri,
      }));

    res.json({
      success: true,
      text,
      sources,
    });
  } catch (error: any) {
    console.error('Error in /api/search-facts:', error?.message || error);
    res.status(500).json({ error: error?.message || 'Gagal mencari fakta iklim' });
  }
});

// API Endpoint 3: Analyze photo for eco-friendly action using gemini-4-argon (with fallback to gemini-3.8-flash)
app.post('/api/analyze-image', async (req: Request, res: Response) => {
  try {
    const { imageBase64, mimeType = 'image/jpeg' } = req.body;
    if (!imageBase64) {
      res.status(400).json({ error: 'Gambar base64 diperlukan' });
      return;
    }

    if (!hasApiKey()) {
      res.json({
        success: true,
        badge: '🌱 Pahlawan Lingkungan Pemula',
        score: 95,
        review: 'Wah, foto aksi lingkunganmu keren sekali! Terima kasih sudah peduli terhadap bumi kita tercinta!',
      });
      return;
    }

    const cleanBase64 = imageBase64.replace(/^data:image\/[a-zA-Z0-9+.-]+;base64,/, '');

    const promptText = `Perhatikan foto yang diunggah oleh siswa SD Kelas 4 ini. Analisislah apakah foto ini berhubungan dengan lingkungan atau aksi ramah bumi (contoh: tanaman, pohon, daur ulang sampah, bersepeda, berjalan kaki, botol minum ulang, mematikan lampu, atau pemandangan alam/langit/polusi). 
Beri respon dalam format JSON dengan properti:
1. "isEcoFriendly": boolean (apakah berhubungan positif dengan lingkungan / kesadaran iklim)
2. "title": judul pujian singkat dan ceria (maksimal 6 kata)
3. "feedback": penjelasan edukatif ramah anak kelas 4 SD tentang apa yang terlihat dan dampaknya bagi bumi (2-3 kalimat hangat)
4. "ecoScore": angka penilaian aksi ramah lingkungan dari 60 hingga 100
5. "badge": gelar anak pahlawan bumi (misal: "🌳 Penjaga Hutan Cilik", "🚴 Juara Transportasi Hijau", "💧 Sahabat Hemat Energi")`;

    let modelToUse = 'gemini-4-argon';
    let response;

    try {
      response = await ai.models.generateContent({
        model: modelToUse,
        contents: {
          parts: [
            {
              inlineData: {
                data: cleanBase64,
                mimeType,
              },
            },
            {
              text: promptText,
            },
          ],
        },
        config: {
          responseMimeType: 'application/json',
        },
      });
    } catch (modelError: any) {
      console.warn('Fallback to gemini-3.8-flash for image analysis:', modelError?.message);
      modelToUse = 'gemini-3.8-flash';
      response = await ai.models.generateContent({
        model: modelToUse,
        contents: {
          parts: [
            {
              inlineData: {
                data: cleanBase64,
                mimeType,
              },
            },
            {
              text: promptText,
            },
          ],
        },
        config: {
          responseMimeType: 'application/json',
        },
      });
    }

    const text = response.text || '{}';
    let parsedData = {};
    try {
      parsedData = JSON.parse(text);
    } catch (e) {
      parsedData = {
        title: 'Aksi Lingkungan Keren!',
        feedback: text,
        ecoScore: 90,
        badge: '🌍 Sahabat Bumi',
      };
    }

    res.json({
      success: true,
      data: parsedData,
    });
  } catch (error: any) {
    console.error('Error in /api/analyze-image:', error?.message || error);
    res.status(500).json({ error: error?.message || 'Gagal menganalisis gambar' });
  }
});

// Setup Vite middleware in dev or static files in production
async function startServer() {
  if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.join(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    });
  } else {
    const { createServer } = await import('vite');
    const vite = await createServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server simulasi IPAS berjalan di http://0.0.0.0:${PORT}`);
  });
}

startServer();
