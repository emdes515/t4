import express from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import axios from 'axios';
import * as cheerio from 'cheerio';
import cors from 'cors';

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());
  app.use(cors());

  // API routes FIRST
  app.post('/api/scrape', async (req, res) => {
    try {
      const { url } = req.body;
      if (!url) {
        return res.status(400).json({ error: 'URL is required' });
      }

      console.log(`Attempting direct scrape for: ${url}`);
      
      // Basic cheerio scraping
      const response = await axios.get(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
          'Accept-Language': 'pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7'
        },
        timeout: 15000
      });
      
      const $ = cheerio.load(response.data);
      // Remove scripts, styles, and other non-content tags
      $('script, style, noscript, iframe, img, svg, header, footer, nav, aside').remove();
      
      // Try to find the main content area first
      let mainContent = $('main, article, [role="main"], .job-description, .offer-description, #job-description, [data-test="section-description"]').text();
      
      // If no main content found, fall back to body
      if (!mainContent || mainContent.trim().length < 100) {
        mainContent = $('body').text();
      }
      
      const text = mainContent.replace(/\s+/g, ' ').trim();
      
      console.log(`Scraped text length: ${text.length}`);
      
      if (!text || text.length < 150) {
        return res.status(400).json({ error: 'Could not extract enough text from page. It might be a SPA or blocked.' });
      }

      res.json({ text });
    } catch (error: any) {
      if (error.response) {
        if (error.response.status === 403 || error.response.status === 401) {
           console.log(`Scraping blocked by target server (Status: ${error.response.status}). This is expected for protected sites.`);
           return res.status(403).json({ error: 'Access denied by the target server (bot protection).' });
        }
        console.error(`Scraping error: ${error.message} (Status: ${error.response.status})`);
      } else {
        console.error('Scraping error:', error.message);
      }
      res.status(500).json({ error: 'Failed to scrape URL' });
    }
  });

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
