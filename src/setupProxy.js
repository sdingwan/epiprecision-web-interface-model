const express = require('express');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const FINAL_DIR = path.resolve(__dirname, '../final');
const PIPELINE_SCRIPT = path.join(FINAL_DIR, 'pipeline.py');
const PIPELINE_RESULTS_JSON = path.join(FINAL_DIR, 'pipeline_results.json');
const ANALYSIS_SUMMARY_JSON = path.join(FINAL_DIR, 'analysis_summary.json');
const DBSCAN_OUTPUT_DIR = path.join(FINAL_DIR, 'dbscan_outputs');
const DBSCAN_METADATA_JSON = path.join(FINAL_DIR, 'dbscan_outputs.json');
const UPLOAD_ROOT = path.join(FINAL_DIR, 'uploads');
const DEFAULT_CNN_OUTPUT = path.join(FINAL_DIR, 'predictions_DL.csv');
const DEFAULT_KL_OUTPUT = path.join(FINAL_DIR, 'predictions_KL.csv');
const REPO_ROOT = path.resolve(__dirname, '..');

fs.mkdirSync(UPLOAD_ROOT, { recursive: true });
fs.mkdirSync(DBSCAN_OUTPUT_DIR, { recursive: true });

let latestUpload = null;

const sanitizeCaseId = (value = '') => {
  const trimmed = (value || '').trim();
  const sanitized = trimmed.replace(/[^A-Za-z0-9-_]/g, '_');
  return sanitized || 'case';
};

const normalizeRelativePath = (relativePath = '') =>
  relativePath.replace(/\\/g, '/').replace(/^\/+/, '');

const walkFiles = (dir, predicate) => {
  const stack = [dir];
  while (stack.length > 0) {
    const current = stack.pop();
    const entries = fs.readdirSync(current, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(current, entry.name);
      if (predicate(entry, fullPath)) {
        return fullPath;
      }
      if (entry.isDirectory()) {
        stack.push(fullPath);
      }
    }
  }
  return null;
};

/**
 * Attempt to run the pipeline with the provided python executable.
 */
const runWithExecutable = (executable, env) =>
  new Promise((resolve, reject) => {
    const child = spawn(executable, [PIPELINE_SCRIPT], {
      cwd: FINAL_DIR,
      env,
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (data) => {
      stdout += data.toString();
      process.stdout.write(data);
    });

    child.stderr.on('data', (data) => {
      stderr += data.toString();
      process.stderr.write(data);
    });

    child.on('error', (error) => {
      reject(error);
    });

    child.on('close', (code) => {
      if (code !== 0) {
        reject(
          new Error(
            `Pipeline exited with code ${code}${stderr ? `: ${stderr}` : ''}`
          )
        );
      } else {
        resolve({ stdout, stderr });
      }
    });
  });

/**
 * Run the Python pipeline, attempting multiple executables until one succeeds.
 */
const runPipeline = async () => {
  const candidates = [
    process.env.PYTHON_PATH,
    'python3',
    'python',
  ].filter(Boolean);

  let lastError = new Error('No python executable candidates found.');

  for (const executable of candidates) {
    try {
      const pipelineEnv = {
        ...process.env,
        CNN_OUTPUT_CSV: DEFAULT_CNN_OUTPUT,
        KNOWLEDGE_OUTPUT_CSV: DEFAULT_KL_OUTPUT,
      };

      if (latestUpload) {
        pipelineEnv.CASE_ROOT_DIR = latestUpload.caseRootDir;
        pipelineEnv.CNN_IMAGE_DIR = latestUpload.imageDir;
        pipelineEnv.KNOWLEDGE_SUBJECT_DIR = latestUpload.caseRootDir;
        pipelineEnv.WORKSPACE_ROOT_DIR = latestUpload.workspaceRootDir;
        if (latestUpload.caseFolderName) {
          pipelineEnv.CASE_ID = sanitizeCaseId(latestUpload.caseFolderName);
        }
        if (latestUpload.workspaceFile) {
          pipelineEnv.WORKSPACE_FILE = latestUpload.workspaceFile;
        } else {
          const workspaceCandidate = path.join(
            latestUpload.workspaceRootDir,
            `Workspace-${latestUpload.caseFolderName}V4.mat`
          );
          if (fs.existsSync(workspaceCandidate)) {
            pipelineEnv.WORKSPACE_FILE = workspaceCandidate;
          }
        }
      }

      return await runWithExecutable(executable, pipelineEnv);
    } catch (error) {
      lastError = error;
      if (error.code === 'ENOENT') {
        continue;
      }
      break;
    }
  }

  throw lastError;
};

const readJsonIfExists = (jsonPath) => {
  if (!fs.existsSync(jsonPath)) {
    return null;
  }
  const raw = fs.readFileSync(jsonPath, 'utf-8');
  return JSON.parse(raw);
};

module.exports = function setupProxy(app) {
  app.use(express.json({ limit: '1024mb' }));
  app.use('/analysis-assets', express.static(DBSCAN_OUTPUT_DIR));

  app.post('/api/upload-case', async (req, res) => {
    try {
      const { files } = req.body || {};
      if (!Array.isArray(files) || files.length === 0) {
        return res.status(400).json({ error: 'No files provided.' });
      }

      const timestamp = Date.now();
      const firstPathEntry = files.find((f) => f.relativePath)?.relativePath || files[0].name;
      const normalizedFirst = normalizeRelativePath(firstPathEntry);
      const topLevelName = normalizedFirst.split('/')[0] || `case-${timestamp}`;
      const uploadBase = path.join(UPLOAD_ROOT, `${topLevelName}-${timestamp}`);

      const caseFolders = new Set();

      for (const file of files) {
        const relativePath = normalizeRelativePath(file.relativePath || file.name);
        if (!file.content) {
          throw new Error(`Missing file content for ${relativePath}`);
        }

        const segments = relativePath.split('/').filter(Boolean);
        if (segments.length === 0) {
          segments.push(file.name);
        }
        caseFolders.add(segments[0]);

        const targetPath = path.join(uploadBase, ...segments);
        if (!targetPath.startsWith(uploadBase)) {
          throw new Error(`Invalid relative path: ${relativePath}`);
        }

        fs.mkdirSync(path.dirname(targetPath), { recursive: true });
        const buffer = Buffer.from(file.content, 'base64');
        fs.writeFileSync(targetPath, buffer);
      }

      const [caseFolderName] = Array.from(caseFolders);
      let caseRootDir =
        caseFolders.size === 1 && caseFolderName
          ? path.join(uploadBase, caseFolderName)
          : uploadBase;

      const reportDir =
        walkFiles(uploadBase, (entry, fullPath) =>
          entry.isDirectory() && entry.name.toLowerCase() === 'report'
        ) || path.join(caseRootDir, 'report');

      let imageDir = reportDir;
      if (!fs.existsSync(imageDir)) {
        throw new Error('Unable to locate report directory in uploaded files.');
      }

      // Determine subject root (folder containing MO or report)
      let subjectRoot = path.dirname(imageDir);
      if (path.basename(subjectRoot).toLowerCase() === 'mo') {
        subjectRoot = path.dirname(subjectRoot);
      }
      caseRootDir = subjectRoot;

      let resolvedCaseName =
        caseFolderName && caseFolderName.toLowerCase() !== 'report'
          ? caseFolderName
          : path.basename(subjectRoot);

      let workspaceFile = walkFiles(uploadBase, (entry, fullPath) =>
        entry.isFile() && /^Workspace-.*V4\.mat$/i.test(entry.name)
      );

      if (!workspaceFile) {
        const repoCandidates = fs
          .readdirSync(REPO_ROOT)
          .filter((name) => /^Workspace-.*V4\.mat$/i.test(name));

        if (repoCandidates.length === 1) {
          workspaceFile = path.join(REPO_ROOT, repoCandidates[0]);
        } else if (repoCandidates.length > 1) {
          const preferred = repoCandidates.find((name) =>
            resolvedCaseName
              ? name.toLowerCase().includes(resolvedCaseName.toLowerCase())
              : false
          );
          workspaceFile = preferred
            ? path.join(REPO_ROOT, preferred)
            : path.join(REPO_ROOT, repoCandidates[0]);
        }
      }

      const workspaceRootDir = workspaceFile
        ? path.dirname(workspaceFile)
        : subjectRoot;

      latestUpload = {
        uploadBase,
        caseRootDir,
        caseFolderName: resolvedCaseName || topLevelName,
        imageDir,
        workspaceRootDir,
        workspaceFile,
      };

      if (
        workspaceFile &&
        latestUpload.caseFolderName &&
        latestUpload.caseFolderName.toLowerCase().startsWith('report')
      ) {
        const match = path
          .basename(workspaceFile)
          .match(/Workspace-(.+?)V4\.mat$/i);
        if (match && match[1]) {
          latestUpload.caseFolderName = match[1];
        }
      }

      res.json({
        fileCount: files.length,
        caseFolderName: latestUpload.caseFolderName,
        caseRootDir: latestUpload.caseRootDir,
        imageDir,
        workspaceFile: latestUpload.workspaceFile || null,
      });
    } catch (error) {
      console.error('Upload error:', error);
      res.status(500).json({
        error: error.message,
      });
    }
  });

  app.post('/api/run-analysis', async (_req, res) => {
    try {
      const execution = await runPipeline();

      const rawResults = readJsonIfExists(PIPELINE_RESULTS_JSON);
      if (!rawResults) {
        throw new Error('Pipeline results JSON not found.');
      }

      const summary = readJsonIfExists(ANALYSIS_SUMMARY_JSON) || {};
      const dbscanMeta = readJsonIfExists(DBSCAN_METADATA_JSON);
      const dbscanMap = new Map();
      if (dbscanMeta?.outputs && Array.isArray(dbscanMeta.outputs)) {
        const expectedCaseId = latestUpload?.caseFolderName
          ? sanitizeCaseId(latestUpload.caseFolderName)
          : null;
        if (!expectedCaseId || dbscanMeta.caseId === expectedCaseId) {
          dbscanMeta.outputs.forEach((item) => {
            if (item?.ic == null || !item?.relative_path) {
              return;
            }
            dbscanMap.set(Number(item.ic), `/analysis-assets/${item.relative_path}`);
          });
        }
      }

      const results = rawResults.map((entry) => ({
        ic: Number(entry.IC),
        dlLabel:
          entry.DL_Label === null || entry.DL_Label === undefined
            ? null
            : Number(entry.DL_Label),
        klPrediction:
          entry.KL_Prediction === null || entry.KL_Prediction === undefined
            ? null
            : Number(entry.KL_Prediction),
        probClass1:
          entry.Prob_Class_1 === null || entry.Prob_Class_1 === undefined
            ? null
            : Number(entry.Prob_Class_1),
        probClass3:
          entry.Prob_Class_3 === null || entry.Prob_Class_3 === undefined
            ? null
            : Number(entry.Prob_Class_3),
        isSoz: Boolean(entry.SOZ),
        reason: entry.Reason,
        explanation: entry.Explanation,
        dbscanImage: dbscanMap.get(Number(entry.IC)) || null,
      }));

      res.json({
        results,
        summary: {
          totalComponents: summary.total_components ?? null,
          sozCount: summary.soz_count ?? 0,
          patientIsSoz: Boolean(summary.patient_is_soz),
          sozIcs: summary.soz_ics ?? [],
        },
        logs: {
          stdout: execution.stdout,
          stderr: execution.stderr,
        },
      });
    } catch (error) {
      res.status(500).json({
        error: error.message,
      });
    }
  });
};
