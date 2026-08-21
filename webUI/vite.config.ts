import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 2007,
    host: true,
    proxy: {
      '/v1': {
        target: 'http://localhost:2005',
        changeOrigin: true,
      },
      '/metrics': {
        target: 'http://localhost:2009',
        changeOrigin: true,
      },
    },
  },
});
