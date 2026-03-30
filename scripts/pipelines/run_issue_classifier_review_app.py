#!/usr/bin/env python3
"""
Serve a local browser app for manual validation of issue-classifier outputs.

The app expects a packet directory created by:

  scripts/pipelines/build_issue_classifier_review_packet.py

It serves a local HTTP UI that lets you:
  - filter to likely printed-law positives
  - inspect the full issue transcript and model evidence
  - accept the model output or override label/operativity
  - record notes and a strict operative-target verdict
  - persist review events and a current-state CSV snapshot
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LEGAL_LABEL_OPTIONS = [
    "none",
    "code_publication_full_issue",
    "code_publication_excerpt_or_installment",
    "amendment_substantial_text",
    "amendment_targeted_text",
    "zoning_ordinance_limited_scope",
    "map_rezoning_order",
    "variance_special_use_order",
    "procedural_notice_only",
    "non_zoning_ordinance",
    "uncertain",
    "unlabeled",
]

OPERATIVITY_OPTIONS = ["", "none", "operative", "proposed", "unclear"]
STRICT_TARGET_OPTIONS = ["", "yes", "no", "unclear"]
REVIEW_DECISION_OPTIONS = ["", "accept", "override", "follow_up"]

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Issue Classifier Review</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #020617;
      --panel: #0f172a;
      --panel-2: #111827;
      --panel-3: #1e293b;
      --border: #334155;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --good: #22c55e;
      --warn: #f59e0b;
      --shadow: rgba(0, 0, 0, 0.26);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
      color: var(--text);
    }
    button, input, select, textarea { font: inherit; }
    a { color: var(--accent); }
    .page {
      max-width: 1500px;
      margin: 0 auto;
      padding: 18px;
      display: grid;
      gap: 14px;
    }
    .card, details {
      background: rgba(15, 23, 42, 0.94);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 14px 40px var(--shadow);
    }
    .card-body, details > div {
      padding: 16px;
    }
    details summary {
      cursor: pointer;
      padding: 16px;
      font-weight: 650;
      color: var(--muted);
    }
    .topbar {
      padding: 16px;
      display: grid;
      gap: 14px;
    }
    .eyebrow {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }
    .headline {
      font-size: 28px;
      font-weight: 750;
      line-height: 1.15;
    }
    .subline {
      color: var(--muted);
      font-size: 14px;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: end;
    }
    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
      min-width: 0;
    }
    input[type="text"], select, textarea {
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid rgba(148, 163, 184, 0.22);
      background: rgba(2, 6, 23, 0.88);
      color: var(--text);
    }
    textarea {
      min-height: 110px;
      resize: vertical;
    }
    button {
      padding: 10px 14px;
      border-radius: 10px;
      border: 1px solid rgba(148, 163, 184, 0.18);
      background: rgba(30, 41, 59, 0.96);
      color: var(--text);
      cursor: pointer;
    }
    button.primary {
      background: linear-gradient(180deg, #0284c7, #0369a1);
      border-color: rgba(56, 189, 248, 0.35);
    }
    button.good {
      background: linear-gradient(180deg, #16a34a, #15803d);
      border-color: rgba(34, 197, 94, 0.35);
    }
    button.warn {
      background: linear-gradient(180deg, #d97706, #b45309);
      border-color: rgba(245, 158, 11, 0.35);
    }
    .button-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .status-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .pill-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(2, 6, 23, 0.9);
      border: 1px solid rgba(148, 163, 184, 0.18);
      font-size: 13px;
    }
    .pill.good { border-color: rgba(34, 197, 94, 0.35); color: #bbf7d0; }
    .pill.warn { border-color: rgba(245, 158, 11, 0.35); color: #fde68a; }
    .grid {
      display: grid;
      grid-template-columns: minmax(360px, 430px) minmax(0, 1fr);
      gap: 14px;
      align-items: start;
    }
    .stack {
      display: grid;
      gap: 14px;
    }
    .section-title {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 10px;
    }
    .prediction-label {
      font-size: clamp(24px, 2.4vw, 34px);
      font-weight: 750;
      line-height: 1.12;
      overflow-wrap: anywhere;
    }
    .prediction-code {
      margin-top: 8px;
      font-size: 12px;
      line-height: 1.4;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace;
      overflow-wrap: anywhere;
    }
    .mini-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }
    .mini-card {
      background: rgba(2, 6, 23, 0.86);
      border: 1px solid rgba(148, 163, 184, 0.14);
      border-radius: 12px;
      padding: 12px;
    }
    .mini-card .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .mini-card .value {
      margin-top: 6px;
      font-size: clamp(16px, 1.7vw, 18px);
      font-weight: 700;
      line-height: 1.2;
      overflow-wrap: anywhere;
    }
    .kv {
      display: grid;
      grid-template-columns: 140px 1fr;
      gap: 10px;
      padding: 10px 0;
      border-bottom: 1px solid rgba(148, 163, 184, 0.08);
    }
    .kv:last-child { border-bottom: none; }
    .kv-key { color: var(--muted); font-size: 13px; }
    .kv-value {
      overflow-wrap: anywhere;
    }
    .transcript {
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace;
      font-size: 12px;
      line-height: 1.5;
      max-height: 68vh;
      overflow: auto;
      border-radius: 12px;
      padding: 14px;
      background: rgba(2, 6, 23, 0.94);
      border: 1px solid rgba(148, 163, 184, 0.16);
    }
    .muted { color: var(--muted); }
    .hidden { display: none !important; }
    .checkbox {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      padding-bottom: 8px;
    }
    .checkbox input { width: auto; }
    .empty {
      padding: 48px;
      text-align: center;
      color: var(--muted);
    }
    @media (max-width: 1100px) {
      .grid {
        grid-template-columns: 1fr;
      }
      .mini-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="card topbar">
      <div>
        <div class="eyebrow">Issue classifier review</div>
        <div id="queueHeadline" class="headline">Loading review packet…</div>
        <div id="queueSubline" class="subline"></div>
      </div>
      <div class="toolbar">
        <label style="min-width:260px;">
          Jump or search
          <input id="jumpInput" type="text" list="issueOptions" placeholder="issue_id, slug, label">
          <datalist id="issueOptions"></datalist>
        </label>
        <label class="checkbox">
          <input id="showReviewedToggle" type="checkbox">
          Show reviewed issues
        </label>
        <div class="button-row">
          <button id="goButton">Go</button>
          <button id="prevButton">Prev</button>
          <button id="nextButton">Next</button>
          <button id="nextUnreviewedButton">Next Unreviewed</button>
          <button id="refreshButton">Refresh</button>
          <a href="/api/export/review_snapshot.csv" target="_blank" rel="noopener">CSV</a>
        </div>
      </div>
    </section>

    <section id="emptyState" class="card empty">Loading review packet…</section>

    <div id="appRoot" class="grid hidden">
      <div class="stack">
        <section class="card">
          <div class="card-body">
            <div class="section-title">Model classification</div>
            <div id="modelLabel" class="prediction-label"></div>
            <div id="modelLabelCode" class="prediction-code"></div>
            <div id="modelMeta" class="muted" style="margin-top:8px;"></div>
            <div id="modelPills" class="pill-row"></div>
            <div class="mini-grid">
              <div class="mini-card">
                <div class="label">Operativity</div>
                <div id="modelOperativity" class="value"></div>
              </div>
              <div class="mini-card">
                <div class="label">Printed-law gate</div>
                <div id="modelPrimaryClass" class="value"></div>
              </div>
              <div class="mini-card">
                <div class="label">Confidence</div>
                <div id="modelConfidence" class="value"></div>
              </div>
            </div>
            <div id="modelEvidence" style="margin-top:14px;"></div>
          </div>
        </section>

        <section class="card">
          <div class="card-body">
            <div class="section-title">Manual decision</div>
            <div id="reviewStatus" class="muted" style="margin-bottom:12px;"></div>
            <div class="button-row" style="margin-bottom:12px;">
              <button id="acceptButton" class="good">Accept prediction</button>
              <button id="fullCodeButton" class="warn">Set full code</button>
              <button id="excerptButton">Set excerpt</button>
            </div>
            <label>
              Manual label
              <select id="manualLabel"></select>
            </label>
            <label style="margin-top:12px;">
              Manual operativity
              <select id="manualOperativity"></select>
            </label>
            <label style="margin-top:12px;">
              Strict operative target
              <select id="strictTarget"></select>
            </label>
            <label style="margin-top:12px;">
              Notes
              <textarea id="noteInput" placeholder="Why the model is right or wrong, especially if this should or should not count as full code."></textarea>
            </label>
            <div class="button-row" style="margin-top:12px;">
              <button id="saveButton" class="primary">Save</button>
              <button id="saveNextButton" class="warn">Save & next</button>
              <button id="clearButton">Clear review</button>
            </div>
          </div>
        </section>
      </div>

      <div class="stack">
        <section class="card">
          <div class="card-body">
            <div class="section-title">Issue transcript</div>
            <div id="transcriptMeta" class="muted" style="margin-bottom:12px;"></div>
            <pre id="transcriptBlock" class="transcript"></pre>
          </div>
        </section>

        <details>
          <summary>Raw model JSON</summary>
          <div>
            <pre id="rawJsonBlock" class="transcript"></pre>
          </div>
        </details>

        <details>
          <summary>Source pages</summary>
          <div>
            <div id="sourcePagesBlock" class="transcript"></div>
          </div>
        </details>
      </div>
    </div>
  </div>
  <script>
    const LABEL_PRIORITY = {
      code_publication_full_issue: 0,
      code_publication_excerpt_or_installment: 1,
      amendment_substantial_text: 2,
      amendment_targeted_text: 3,
      zoning_ordinance_limited_scope: 4,
      map_rezoning_order: 5,
      variance_special_use_order: 6,
      procedural_notice_only: 7,
      non_zoning_ordinance: 8,
      none: 9,
      uncertain: 10,
      unlabeled: 11,
    };

    const state = {
      config: null,
      items: [],
      queue: [],
      selectedIssueId: null,
      currentItem: null,
      saving: false,
    };

    function prettyNumber(value) {
      if (value === null || value === undefined || value === '') return '—';
      return new Intl.NumberFormat().format(Number(value));
    }

    function confidenceText(value) {
      if (value === null || value === undefined || value === '') return '—';
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) return String(value);
      return numeric.toFixed(2);
    }

    function formatEnumLabel(value) {
      const raw = String(value ?? '').trim();
      if (!raw) return '—';
      return raw
        .replaceAll('_', ' ')
        .replaceAll('-', ' ')
        .replace(/\\s+/g, ' ')
        .replace(/\\b\\w/g, character => character.toUpperCase());
    }

    function escapeHtml(value) {
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('\"', '&quot;');
    }

    function fetchJson(path, options = {}) {
      return fetch(path, options).then(async response => {
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || `HTTP ${response.status}`);
        }
        return response.json();
      });
    }

    function activeReview(summary) {
      return summary.review || null;
    }

    function queueSort(left, right) {
      const leftReview = left.review ? 1 : 0;
      const rightReview = right.review ? 1 : 0;
      if (leftReview !== rightReview) return leftReview - rightReview;
      const leftPriority = LABEL_PRIORITY[left.predicted_label] ?? 999;
      const rightPriority = LABEL_PRIORITY[right.predicted_label] ?? 999;
      if (leftPriority !== rightPriority) return leftPriority - rightPriority;
      const confidenceDelta = Number(right.confidence_0_to_1 || 0) - Number(left.confidence_0_to_1 || 0);
      if (confidenceDelta !== 0) return confidenceDelta;
      return left.issue_id.localeCompare(right.issue_id);
    }

    function rebuildQueue() {
      const search = document.getElementById('jumpInput').value.trim().toLowerCase();
      const showReviewed = document.getElementById('showReviewedToggle').checked;
      let items = state.items.filter(item => showReviewed || !item.review);
      if (search) {
        items = items.filter(item => {
          const haystack = [
            item.issue_id,
            item.slug,
            item.predicted_label,
            item.predicted_operativity,
            item.preview,
          ].join(' ').toLowerCase();
          return haystack.includes(search);
        });
      }
      state.queue = [...items].sort(queueSort);
      if (!state.queue.length) {
        state.selectedIssueId = null;
        state.currentItem = null;
      } else if (!state.selectedIssueId || !state.queue.some(item => item.issue_id === state.selectedIssueId)) {
        state.selectedIssueId = state.queue[0].issue_id;
      }
      renderQueueMeta();
    }

    function renderQueueMeta() {
      const total = state.items.length;
      const reviewed = state.items.filter(item => item.review).length;
      const queueIndex = state.queue.findIndex(item => item.issue_id === state.selectedIssueId);
      const queuePosition = queueIndex >= 0 ? `${queueIndex + 1}/${state.queue.length}` : `0/${state.queue.length}`;
      document.getElementById('queueHeadline').textContent = state.selectedIssueId || 'No issues in the current queue';
      const filterNotes = [];
      if (state.config?.metadata?.only_zoning_legal_text) filterNotes.push('printed zoning law only');
      if (state.config?.metadata?.exclude_proposed) filterNotes.push('model-proposed items excluded');
      const filterText = filterNotes.length ? ` · ${filterNotes.join(' · ')}` : '';
      document.getElementById('queueSubline').textContent = `${queuePosition} in queue · ${reviewed}/${total} reviewed${filterText}`;
      document.title = state.selectedIssueId ? `Issue Review · ${queuePosition} · ${state.selectedIssueId}` : 'Issue Review';
      document.getElementById('emptyState').textContent = state.queue.length
        ? 'Loading selected issue…'
        : 'No issues match the current queue settings.';
    }

    function populateSelect(elementId, values, { includeBlank = false } = {}) {
      const element = document.getElementById(elementId);
      const options = [];
      if (includeBlank) {
        options.push('<option value="">—</option>');
      }
      for (const value of values) {
        options.push(`<option value="${escapeHtml(value)}">${escapeHtml(formatEnumLabel(value))}</option>`);
      }
      element.innerHTML = options.join('');
    }

    function populateIssueOptions() {
      const dataList = document.getElementById('issueOptions');
      dataList.innerHTML = state.items.map(item => `<option value="${escapeHtml(item.issue_id)}"></option>`).join('');
    }

    async function loadConfig() {
      state.config = await fetchJson('/api/config');
      populateSelect('manualLabel', state.config.legal_label_options);
      populateSelect('manualOperativity', state.config.manual_operativity_options.filter(Boolean), { includeBlank: true });
      populateSelect('strictTarget', state.config.strict_target_options.filter(Boolean), { includeBlank: true });
    }

    async function loadItems() {
      const payload = await fetchJson('/api/items');
      state.items = payload.items;
      populateIssueOptions();
      rebuildQueue();
      if (state.selectedIssueId) {
        await selectIssue(state.selectedIssueId, { preserveQueue: true });
      } else {
        document.getElementById('appRoot').classList.add('hidden');
        document.getElementById('emptyState').classList.remove('hidden');
      }
    }

    function setQuickLabel(value) {
      document.getElementById('manualLabel').value = value;
      document.getElementById('reviewStatus').textContent = `Manual label set to ${value}`;
    }

    async function selectIssue(issueId, { preserveQueue = false } = {}) {
      if (!preserveQueue) {
        state.selectedIssueId = issueId;
        rebuildQueue();
      } else {
        state.selectedIssueId = issueId;
        renderQueueMeta();
      }
      if (!state.selectedIssueId) return;
      const payload = await fetchJson(`/api/items/${encodeURIComponent(state.selectedIssueId)}`);
      state.currentItem = payload.item;
      const review = payload.review || null;
      const predicted = payload.item.predicted || {};

      document.getElementById('emptyState').classList.add('hidden');
      document.getElementById('appRoot').classList.remove('hidden');

      document.getElementById('modelLabel').textContent = formatEnumLabel(predicted.label || 'unlabeled');
      document.getElementById('modelLabelCode').textContent = predicted.label || 'unlabeled';
      document.getElementById('modelMeta').textContent = `${payload.item.issue_date || 'unknown date'} · ${payload.item.slug || 'unknown slug'} · ${prettyNumber(payload.item.issue_chars)} chars`;
      document.getElementById('modelOperativity').textContent = formatEnumLabel(predicted.operativity);
      document.getElementById('modelPrimaryClass').textContent = formatEnumLabel(predicted.primary_class);
      document.getElementById('modelConfidence').textContent = confidenceText(predicted.confidence_0_to_1);
      const pills = [
        predicted.legal_action && `<span class="pill">${escapeHtml(formatEnumLabel(predicted.legal_action))}</span>`,
        predicted.scope && `<span class="pill">${escapeHtml(formatEnumLabel(predicted.scope))}</span>`,
        predicted.publication_completeness && predicted.publication_completeness !== 'none' ? `<span class="pill">${escapeHtml(formatEnumLabel(predicted.publication_completeness))}</span>` : '',
        ...(Array.isArray(predicted.quality_flags) ? predicted.quality_flags.map(flag => `<span class="pill warn">${escapeHtml(formatEnumLabel(flag))}</span>`) : []),
      ].filter(Boolean);
      document.getElementById('modelPills').innerHTML = pills.join('');
      const evidenceRows = [
        ['Zoning evidence', predicted.zoning_evidence_quote],
        ['Legal evidence', predicted.legal_evidence_quote],
        ['Rationale', predicted.rationale],
      ];
      document.getElementById('modelEvidence').innerHTML = evidenceRows.map(([label, value]) => `
        <div class="kv">
          <div class="kv-key">${escapeHtml(label)}</div>
          <div class="kv-value">${escapeHtml(value || '—')}</div>
        </div>
      `).join('');

      document.getElementById('transcriptMeta').textContent = `${prettyNumber(payload.item.page_count)} pages · ${prettyNumber(payload.item.issue_chars)} chars`;
      document.getElementById('transcriptBlock').textContent = payload.item.issue_transcript || '';
      document.getElementById('rawJsonBlock').textContent = JSON.stringify(payload.item.model_output || {}, null, 2);
      const sourcePages = payload.item.source_pages_full || [];
      document.getElementById('sourcePagesBlock').textContent = sourcePages.length
        ? sourcePages.map(page => `[p${page.page_num}] ${page.page_id} · ${page.text_source}\\n${page.source_path}`).join('\\n\\n')
        : 'This packet was built from an issue_txt_dir transcript, so page-level source text was not materialized separately.';

      document.getElementById('manualLabel').value = review?.manual_label || predicted.label || 'unlabeled';
      document.getElementById('manualOperativity').value = review?.manual_operativity || predicted.operativity || '';
      document.getElementById('strictTarget').value = review?.strict_operative_target || '';
      document.getElementById('noteInput').value = review?.note || '';
      document.getElementById('reviewStatus').textContent = review
        ? `Reviewed ${review.reviewed_at} · ${review.review_decision || 'reviewed'}`
        : 'Unreviewed';
    }

    function currentIndex() {
      return state.queue.findIndex(item => item.issue_id === state.selectedIssueId);
    }

    function nextIssueId(direction) {
      if (!state.queue.length) return null;
      const index = currentIndex();
      if (index === -1) return state.queue[0].issue_id;
      return state.queue[(index + direction + state.queue.length) % state.queue.length].issue_id;
    }

    function nextUnreviewedIssueId() {
      for (const item of state.queue) {
        if (!item.review && item.issue_id !== state.selectedIssueId) return item.issue_id;
      }
      return null;
    }

    async function navigate(direction) {
      const issueId = nextIssueId(direction);
      if (issueId) await selectIssue(issueId);
    }

    async function saveReview({ moveNext = false, acceptPrediction = false } = {}) {
      if (!state.currentItem || state.saving) return;
      state.saving = true;
      document.getElementById('reviewStatus').textContent = 'Saving…';
      try {
        const predicted = state.currentItem.predicted || {};
        if (acceptPrediction) {
          document.getElementById('manualLabel').value = predicted.label || 'unlabeled';
          document.getElementById('manualOperativity').value = predicted.operativity || '';
        }
        const payload = {
          manual_label: document.getElementById('manualLabel').value,
          manual_operativity: document.getElementById('manualOperativity').value,
          strict_operative_target: document.getElementById('strictTarget').value,
          note: document.getElementById('noteInput').value,
        };
        await fetchJson(`/api/reviews/${encodeURIComponent(state.currentItem.issue_id)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        await loadItems();
        await selectIssue(state.currentItem.issue_id, { preserveQueue: true });
        if (moveNext) {
          const issueId = nextUnreviewedIssueId() || nextIssueId(1);
          if (issueId) await selectIssue(issueId);
        }
      } catch (error) {
        document.getElementById('reviewStatus').textContent = error.message || String(error);
      } finally {
        state.saving = false;
      }
    }

    async function clearReview() {
      if (!state.currentItem || state.saving) return;
      state.saving = true;
      document.getElementById('reviewStatus').textContent = 'Clearing…';
      try {
        await fetchJson(`/api/reviews/${encodeURIComponent(state.currentItem.issue_id)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ clear: true }),
        });
        await loadItems();
        if (state.selectedIssueId) {
          await selectIssue(state.selectedIssueId, { preserveQueue: true });
        }
      } catch (error) {
        document.getElementById('reviewStatus').textContent = error.message || String(error);
      } finally {
        state.saving = false;
      }
    }

    async function jumpToMatch() {
      rebuildQueue();
      if (state.queue.length) {
        await selectIssue(state.queue[0].issue_id, { preserveQueue: true });
      }
    }

    function wireUi() {
      document.getElementById('showReviewedToggle').addEventListener('change', () => jumpToMatch().catch(showFatal));
      document.getElementById('jumpInput').addEventListener('change', () => jumpToMatch().catch(showFatal));
      document.getElementById('goButton').addEventListener('click', () => jumpToMatch().catch(showFatal));
      document.getElementById('refreshButton').addEventListener('click', () => loadItems().catch(showFatal));
      document.getElementById('prevButton').addEventListener('click', () => navigate(-1));
      document.getElementById('nextButton').addEventListener('click', () => navigate(1));
      document.getElementById('nextUnreviewedButton').addEventListener('click', async () => {
        const issueId = nextUnreviewedIssueId();
        if (issueId) await selectIssue(issueId);
      });
      document.getElementById('acceptButton').addEventListener('click', () => saveReview({ acceptPrediction: true, moveNext: true }));
      document.getElementById('fullCodeButton').addEventListener('click', () => setQuickLabel('code_publication_full_issue'));
      document.getElementById('excerptButton').addEventListener('click', () => setQuickLabel('code_publication_excerpt_or_installment'));
      document.getElementById('saveButton').addEventListener('click', () => saveReview());
      document.getElementById('saveNextButton').addEventListener('click', () => saveReview({ moveNext: true }));
      document.getElementById('clearButton').addEventListener('click', clearReview);
      window.addEventListener('keydown', async event => {
        if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') {
          event.preventDefault();
          await saveReview();
          return;
        }
        if (event.target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)) return;
        if (event.key === 'j') {
          event.preventDefault();
          await navigate(1);
        } else if (event.key === 'k') {
          event.preventDefault();
          await navigate(-1);
        } else if (event.key === 'a') {
          event.preventDefault();
          await saveReview({ acceptPrediction: true, moveNext: true });
        } else if (event.key === 'f') {
          event.preventDefault();
          setQuickLabel('code_publication_full_issue');
        } else if (event.key === 'e') {
          event.preventDefault();
          setQuickLabel('code_publication_excerpt_or_installment');
        }
      });
    }

    function showFatal(error) {
      document.getElementById('appRoot').classList.add('hidden');
      document.getElementById('emptyState').classList.remove('hidden');
      document.getElementById('emptyState').textContent = error.message || String(error);
    }

    async function boot() {
      try {
        await loadConfig();
        wireUi();
        await loadItems();
      } catch (error) {
        showFatal(error);
      }
    }

    boot();
  </script>
</body>
</html>
"""


def _json_dumps(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _norm(value: Any) -> str:
    return str(value or "").strip()


class ReviewAppState:
    def __init__(self, packet_dir: Path) -> None:
        self.packet_dir = packet_dir.resolve()
        self.metadata_path = self.packet_dir / "metadata.json"
        self.index_path = self.packet_dir / "index.jsonl"
        self.review_events_path = self.packet_dir / "review_events.jsonl"
        self.review_snapshot_path = self.packet_dir / "review_snapshot.csv"
        self.lock = threading.Lock()
        self.metadata = self._load_metadata()
        self.item_summaries = self._load_index()
        self.item_path_by_issue = {item["issue_id"]: Path(item["path"]) for item in self.item_summaries}
        self.reviews_by_issue = self._load_reviews()
        self._write_review_snapshot()

    def _load_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.is_file():
            raise SystemExit(f"Missing metadata.json in packet dir: {self.metadata_path}")
        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(f"Invalid metadata JSON: {self.metadata_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise SystemExit(f"metadata.json is not an object: {self.metadata_path}")
        return payload

    def _load_index(self) -> list[dict[str, Any]]:
        if not self.index_path.is_file():
            raise SystemExit(f"Missing index.jsonl in packet dir: {self.index_path}")
        items: list[dict[str, Any]] = []
        with self.index_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception as exc:
                    raise SystemExit(f"Invalid JSON in {self.index_path}:{line_number}: {exc}") from exc
                if not isinstance(payload, dict):
                    raise SystemExit(f"Expected JSON object in {self.index_path}:{line_number}")
                issue_id = _norm(payload.get("issue_id"))
                path_value = _norm(payload.get("path"))
                relative_path_value = _norm(payload.get("relative_path"))
                if not issue_id or not path_value:
                    raise SystemExit(f"Invalid index row in {self.index_path}:{line_number}")
                resolved_path = self._resolve_item_path(
                    raw_path=path_value,
                    relative_path=relative_path_value,
                )
                item = {
                    "issue_id": issue_id,
                    "issue_date": _norm(payload.get("issue_date")),
                    "slug": _norm(payload.get("slug")),
                    "predicted_label": _norm(payload.get("predicted_label")),
                    "predicted_operativity": _norm(payload.get("predicted_operativity")),
                    "predicted_primary_class": _norm(payload.get("predicted_primary_class")),
                    "confidence_0_to_1": payload.get("confidence_0_to_1"),
                    "quality_flags": payload.get("quality_flags") if isinstance(payload.get("quality_flags"), list) else [],
                    "page_count": int(payload.get("page_count") or 0),
                    "issue_chars": int(payload.get("issue_chars") or 0),
                    "preview": _norm(payload.get("preview")),
                    "path": str(resolved_path),
                }
                items.append(item)
        if not items:
            raise SystemExit(f"No review items found in {self.index_path}")
        return items

    def _resolve_item_path(self, *, raw_path: str, relative_path: str) -> Path:
        if relative_path:
            candidate = (self.packet_dir / relative_path).expanduser().resolve()
            if candidate.is_file():
                return candidate
        absolute = Path(raw_path).expanduser().resolve()
        if absolute.is_file():
            return absolute
        fallback = self.packet_dir / absolute.parent.name / absolute.name
        if fallback.is_file():
            return fallback.resolve()
        raise SystemExit(
            "Index row points to a missing review item file and no local fallback matched: "
            f"raw_path={raw_path} relative_path={relative_path or '<none>'}"
        )

    def _load_reviews(self) -> dict[str, dict[str, Any]]:
        reviews: dict[str, dict[str, Any]] = {}
        if not self.review_events_path.exists():
            self.review_events_path.touch()
            return reviews
        with self.review_events_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception as exc:
                    raise SystemExit(f"Invalid review event JSON in {self.review_events_path}:{line_number}: {exc}") from exc
                if not isinstance(payload, dict):
                    raise SystemExit(f"Expected object review event in {self.review_events_path}:{line_number}")
                issue_id = _norm(payload.get("issue_id"))
                if not issue_id:
                    raise SystemExit(f"Review event missing issue_id in {self.review_events_path}:{line_number}")
                if payload.get("event_type") == "clear":
                    reviews.pop(issue_id, None)
                    continue
                review = payload.get("review")
                if not isinstance(review, dict):
                    raise SystemExit(f"Review event missing review object in {self.review_events_path}:{line_number}")
                reviews[issue_id] = review
        return reviews

    def config_payload(self) -> dict[str, Any]:
        label_options = sorted({item["predicted_label"] for item in self.item_summaries if item["predicted_label"]})
        operativity_options = sorted({item["predicted_operativity"] for item in self.item_summaries if item["predicted_operativity"]})
        return {
            "metadata": self.metadata,
            "label_options": label_options,
            "operativity_options": [""] + operativity_options,
            "legal_label_options": LEGAL_LABEL_OPTIONS,
            "manual_operativity_options": OPERATIVITY_OPTIONS,
            "review_decision_options": REVIEW_DECISION_OPTIONS,
            "strict_target_options": STRICT_TARGET_OPTIONS,
        }

    def list_items_payload(self) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        for summary in self.item_summaries:
            merged = dict(summary)
            review = self.reviews_by_issue.get(summary["issue_id"])
            if review:
                merged["review"] = review
            items.append(merged)
        return {"items": items}

    def issue_payload(self, issue_id: str) -> dict[str, Any]:
        issue_key = _norm(issue_id)
        path = self.item_path_by_issue.get(issue_key)
        if path is None or not path.is_file():
            raise KeyError(issue_key)
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(f"Invalid item JSON for issue_id={issue_key}: {path}: {exc}") from exc
        if not isinstance(item, dict):
            raise SystemExit(f"Item file is not a JSON object: {path}")
        review = self.reviews_by_issue.get(issue_key)
        return {"item": item, "review": review}

    def save_review(self, issue_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        issue_key = _norm(issue_id)
        if issue_key not in self.item_path_by_issue:
            raise KeyError(issue_key)
        if payload.get("clear"):
            event = {
                "event_type": "clear",
                "issue_id": issue_key,
                "saved_at": self._timestamp_now(),
            }
            with self.lock:
                self._append_review_event(event)
                self.reviews_by_issue.pop(issue_key, None)
                self._write_review_snapshot()
            return {"cleared": True}

        manual_label = _norm(payload.get("manual_label"))
        manual_operativity = _norm(payload.get("manual_operativity"))
        review_decision = _norm(payload.get("review_decision"))
        strict_operative_target = _norm(payload.get("strict_operative_target"))
        note = _norm(payload.get("note"))
        if manual_label and manual_label not in LEGAL_LABEL_OPTIONS:
            raise ValueError(f"Invalid manual_label: {manual_label}")
        if manual_operativity not in OPERATIVITY_OPTIONS:
            raise ValueError(f"Invalid manual_operativity: {manual_operativity}")
        if review_decision not in REVIEW_DECISION_OPTIONS:
            raise ValueError(f"Invalid review_decision: {review_decision}")
        if strict_operative_target not in STRICT_TARGET_OPTIONS:
            raise ValueError(f"Invalid strict_operative_target: {strict_operative_target}")
        item_payload = self.issue_payload(issue_key)["item"]
        predicted = item_payload.get("predicted") or {}
        if not review_decision:
            predicted_label = _norm(predicted.get("label"))
            predicted_operativity = _norm(predicted.get("operativity"))
            if manual_label == predicted_label and manual_operativity == predicted_operativity:
                review_decision = "accept"
            else:
                review_decision = "override"
        exact_label_match = ""
        exact_operativity_match = ""
        if manual_label:
            exact_label_match = "yes" if manual_label == _norm(predicted.get("label")) else "no"
        if manual_operativity:
            exact_operativity_match = "yes" if manual_operativity == _norm(predicted.get("operativity")) else "no"
        review = {
            "issue_id": issue_key,
            "manual_label": manual_label,
            "manual_operativity": manual_operativity,
            "review_decision": review_decision,
            "strict_operative_target": strict_operative_target,
            "exact_label_match": exact_label_match,
            "exact_operativity_match": exact_operativity_match,
            "note": note,
            "reviewed_at": self._timestamp_now(),
        }
        event = {
            "event_type": "upsert",
            "issue_id": issue_key,
            "saved_at": review["reviewed_at"],
            "review": review,
        }
        with self.lock:
            self._append_review_event(event)
            self.reviews_by_issue[issue_key] = review
            self._write_review_snapshot()
        return {"review": review}

    def _append_review_event(self, payload: dict[str, Any]) -> None:
        self.review_events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.review_events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_review_snapshot(self) -> None:
        fieldnames = [
            "issue_id",
            "predicted_label",
            "predicted_operativity",
            "manual_label",
            "manual_operativity",
            "review_decision",
            "strict_operative_target",
            "exact_label_match",
            "exact_operativity_match",
            "note",
            "reviewed_at",
            "path",
        ]
        self.review_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with self.review_snapshot_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for summary in sorted(self.item_summaries, key=lambda row: row["issue_id"]):
                review = self.reviews_by_issue.get(summary["issue_id"])
                if not review:
                    continue
                writer.writerow(
                    {
                        "issue_id": summary["issue_id"],
                        "predicted_label": summary["predicted_label"],
                        "predicted_operativity": summary["predicted_operativity"],
                        "manual_label": review.get("manual_label", ""),
                        "manual_operativity": review.get("manual_operativity", ""),
                        "review_decision": review.get("review_decision", ""),
                        "strict_operative_target": review.get("strict_operative_target", ""),
                        "exact_label_match": review.get("exact_label_match", ""),
                        "exact_operativity_match": review.get("exact_operativity_match", ""),
                        "note": review.get("note", ""),
                        "reviewed_at": review.get("reviewed_at", ""),
                        "path": summary["path"],
                    }
                )

    @staticmethod
    def _timestamp_now() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()


class ReviewRequestHandler(BaseHTTPRequestHandler):
    server: "ReviewHTTPServer"

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        if parsed_url.path == "/favicon.ico":
            self._send_bytes(b"", content_type="image/x-icon", status=HTTPStatus.NO_CONTENT)
            return
        if parsed_url.path == "/":
            self._send_html(INDEX_HTML)
            return
        if parsed_url.path == "/api/config":
            self._send_json(self.server.state.config_payload())
            return
        if parsed_url.path == "/api/items":
            self._send_json(self.server.state.list_items_payload())
            return
        if parsed_url.path.startswith("/api/items/"):
            issue_id = parsed_url.path.removeprefix("/api/items/")
            try:
                payload = self.server.state.issue_payload(issue_id)
            except KeyError:
                self._send_error_text(HTTPStatus.NOT_FOUND, f"Unknown issue_id: {issue_id}")
                return
            self._send_json(payload)
            return
        if parsed_url.path == "/api/export/review_snapshot.csv":
            snapshot_path = self.server.state.review_snapshot_path
            self._send_bytes(
                snapshot_path.read_bytes(),
                content_type="text/csv; charset=utf-8",
                status=HTTPStatus.OK,
            )
            return
        self._send_error_text(HTTPStatus.NOT_FOUND, f"Unknown route: {html.escape(parsed_url.path)}")

    def do_POST(self) -> None:
        parsed_url = urlparse(self.path)
        if not parsed_url.path.startswith("/api/reviews/"):
            self._send_error_text(HTTPStatus.NOT_FOUND, f"Unknown route: {html.escape(parsed_url.path)}")
            return
        issue_id = parsed_url.path.removeprefix("/api/reviews/")
        content_length = int(self.headers.get("Content-Length") or 0)
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception as exc:
            self._send_error_text(HTTPStatus.BAD_REQUEST, f"Invalid JSON body: {exc}")
            return
        if not isinstance(payload, dict):
            self._send_error_text(HTTPStatus.BAD_REQUEST, "JSON body must be an object")
            return
        try:
            result = self.server.state.save_review(issue_id, payload)
        except KeyError:
            self._send_error_text(HTTPStatus.NOT_FOUND, f"Unknown issue_id: {issue_id}")
            return
        except ValueError as exc:
            self._send_error_text(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._send_json(result)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(_json_dumps(payload), content_type="application/json; charset=utf-8", status=status)

    def _send_html(self, payload: str, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(payload.encode("utf-8"), content_type="text/html; charset=utf-8", status=status)

    def _send_error_text(self, status: HTTPStatus, message: str) -> None:
        self._send_bytes(message.encode("utf-8"), content_type="text/plain; charset=utf-8", status=status)

    def _send_bytes(self, payload: bytes, *, content_type: str, status: HTTPStatus) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)


class ReviewHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler: type[BaseHTTPRequestHandler], state: ReviewAppState):
        super().__init__(server_address, handler)
        self.state = state


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local browser review app for issue-classifier outputs.")
    parser.add_argument("--packet-dir", required=True, help="Review packet directory built by build_issue_classifier_review_packet.py")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8787, help="Port to bind.")
    parser.add_argument("--open-browser", action="store_true", help="Open the local review app in the default browser.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    packet_dir = Path(args.packet_dir).expanduser().resolve()
    if not packet_dir.is_dir():
        raise SystemExit(f"--packet-dir is not a directory: {packet_dir}")
    state = ReviewAppState(packet_dir)
    server = ReviewHTTPServer((args.host, args.port), ReviewRequestHandler, state)
    url = f"http://{args.host}:{args.port}"
    print(f"review app serving {url}", flush=True)
    print(f"packet_dir={packet_dir}", flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
