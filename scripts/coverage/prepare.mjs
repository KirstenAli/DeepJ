import { existsSync, unlinkSync } from 'node:fs';

for (const file of ['target/jacoco.exec', 'target/site/jacoco/jacoco.xml']) {
  if (existsSync(file)) unlinkSync(file);
}
