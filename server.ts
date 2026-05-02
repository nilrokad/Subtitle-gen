import express from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import multer from 'multer';
import { AssemblyAI } from 'assemblyai';
import fs from 'fs';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const PORT = 3000;

// Configure Multer for file uploads
const upload = multer({ 
  dest: '/tmp/uploads/',
  limits: { fileSize: 200 * 1024 * 1024 } // 200MB limit
});

// Initialize AssemblyAI
const client = new AssemblyAI({
  apiKey: process.env.ASSEMBLYAI_API_KEY || '',
});

async function startServer() {
  app.use(express.json());

  // API Endpoint to start transcription
  app.post('/api/transcribe', upload.single('audio'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No audio file uploaded' });
      }

      if (!process.env.ASSEMBLYAI_API_KEY) {
        return res.status(500).json({ error: 'ASSEMBLYAI_API_KEY is not set in environment variables. Please add it in the Settings menu.' });
      }

      const filePath = req.file.path;
      const fileName = req.file.originalname;
      console.log('Starting transcription for:', fileName);

      try {
        // Just start the transcription and return the ID
        const transcript = await client.transcripts.submit({
          audio: filePath,
          speech_models: ['universal-3-pro', 'universal-2'],
          language_detection: true,
        });

        res.json({ id: transcript.id, fileName });
      } finally {
        // Clean up uploaded file
        if (fs.existsSync(filePath)) {
          fs.unlinkSync(filePath);
        }
      }
    } catch (error: any) {
      console.error('Transcription error:', error);
      res.status(500).json({ error: error.message || 'Internal server error' });
    }
  });

  // API Endpoint to check transcription status
  app.get('/api/status/:id', async (req, res) => {
    try {
      const { id } = req.params;
      const transcript = await client.transcripts.get(id);

      if (transcript.status === 'error') {
        return res.json({ status: 'error', error: transcript.error });
      }

      if (transcript.status === 'completed') {
        // Fetch sentences and paragraphs for more context
        const [{ sentences }, { paragraphs }] = await Promise.all([
          client.transcripts.sentences(id),
          client.transcripts.paragraphs(id)
        ]);

        return res.json({
          status: 'completed',
          result: {
            id: transcript.id,
            text: transcript.text,
            words: transcript.words,
            sentences,
            paragraphs
          }
        });
      }

      res.json({ status: transcript.status });
    } catch (error: any) {
      console.error('Status check error:', error);
      res.status(500).json({ error: error.message || 'Internal server error' });
    }
  });

  // Error handler
  app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
    console.error('Unhandled error:', err);
    res.status(500).json({ error: err.message || 'Internal server error' });
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
