import '@/styles/globals.css';
import type { AppProps } from 'next/app';
import { SpotlightBackground } from '@/components/ui/spotlight-background';

export default function App({ Component, pageProps }: AppProps) {
  return (
    <SpotlightBackground
      spotlightSize={500}
      spotlightClassName="from-aurora-purple/40 via-aurora-blue/30 to-transparent"
    >
      <Component {...pageProps} />
    </SpotlightBackground>
  );
}