const storageKey = 'maplescreen.keywordAlerts';

const normalize = text => text.normalize('NFKC').toLowerCase();

function parseKeywords(input) {
  const keywords = new Map();
  for (const item of input.split(/[,，\r\n]+/)) {
    const keyword = item.trim();
    const normalized = normalize(keyword);
    if (keyword && !keywords.has(normalized)) keywords.set(normalized, keyword);
  }
  return [...keywords.values()];
}

export function createKeywordAlerts() {
  const panel = document.getElementById('keyword-alerts');
  const form = document.getElementById('alert-form');
  const keywordsInput = document.getElementById('alert-keywords');
  const soundInput = document.getElementById('alert-sound');
  const popupInput = document.getElementById('alert-popup');
  const status = document.getElementById('alert-status');
  const toast = document.getElementById('alert-toast');
  const toastText = document.getElementById('alert-toast-text');

  let settings = { keywords: [], sound: false, popup: false };
  try {
    const saved = JSON.parse(window.localStorage.getItem(storageKey));
    if (Array.isArray(saved?.keywords)) {
      settings = {
        keywords: parseKeywords(saved.keywords.filter(word => typeof word === 'string').join('\n')),
        sound: saved.sound === true,
        popup: saved.popup === true,
      };
    }
  } catch {
    // Reminders can still run for the current page if storage is unavailable.
  }

  let audio;
  let nextSoundAt = 0;
  let statusVersion = 0;
  let sessionId;
  let lastSeen = 0;

  function showStatus(message, options = settings) {
    const parts = [message];
    if (options.sound) {
      if (!(window.AudioContext || window.webkitAudioContext)) parts.push('当前浏览器不支持提示音');
      else parts.push(audio?.state === 'running' ? '提示音已就绪' : '点击保存或测试以启用提示音');
    }
    if (options.popup) {
      parts.push(window.Notification?.permission === 'granted' ? '后台可用系统通知' : '弹窗在页面内显示');
    }
    status.textContent = parts.join('。') + '。';
  }

  async function enableSound() {
    const Audio = window.AudioContext || window.webkitAudioContext;
    if (!Audio) return;
    try {
      audio ??= new Audio();
      if (audio.state !== 'running') await audio.resume();
    } catch {
      // The settings status explains when browser audio still needs activation.
    }
  }

  async function enableSystemPopup() {
    if (window.Notification?.permission !== 'default') return;
    try {
      await window.Notification.requestPermission();
    } catch {
      // The in-page popup works even when the browser disallows permission prompts.
    }
  }

  function beep() {
    if (audio?.state !== 'running' || audio.currentTime < nextSoundAt) return;
    const start = audio.currentTime;
    nextSoundAt = start + 0.45;
    const oscillator = audio.createOscillator();
    const volume = audio.createGain();
    oscillator.frequency.setValueAtTime(660, start);
    oscillator.frequency.setValueAtTime(880, start + 0.10);
    volume.gain.setValueAtTime(0, start);
    volume.gain.linearRampToValueAtTime(0.12, start + 0.02);
    volume.gain.linearRampToValueAtTime(0, start + 0.23);
    oscillator.connect(volume).connect(audio.destination);
    oscillator.start(start);
    oscillator.stop(start + 0.23);
  }

  function popup(message) {
    if ((document.hidden || !document.hasFocus()) && window.Notification?.permission === 'granted') {
      try {
        new window.Notification('MapleScreen 关键词提醒', {
          body: message, tag: 'maplescreen-keyword-alert', silent: true,
        });
        return;
      } catch {
        // A few browsers expose Notification but disallow direct construction.
      }
    }
    toastText.textContent = message;
    toast.hidden = false;
  }

  function prepare(options, message, testSound = false) {
    const version = ++statusVersion;
    const refresh = () => {
      if (version === statusVersion) showStatus(message, options);
    };
    refresh();
    if (options.sound) enableSound().then(() => {
      if (testSound && version === statusVersion) beep();
      refresh();
    });
    if (options.popup) enableSystemPopup().then(refresh);
  }

  keywordsInput.value = settings.keywords.join('\n');
  soundInput.checked = settings.sound;
  popupInput.checked = settings.popup;
  panel.hidden = false;
  if (settings.keywords.length && (settings.sound || settings.popup)) {
    showStatus(`已保存 ${settings.keywords.length} 个关键词`);
  } else {
    showStatus(settings.keywords.length ? '关键词提醒已关闭' : '请输入关键词', {});
  }

  form.addEventListener('submit', event => {
    event.preventDefault();
    settings = {
      keywords: parseKeywords(keywordsInput.value), sound: soundInput.checked, popup: popupInput.checked,
    };
    const enabled = settings.keywords.length > 0 && (settings.sound || settings.popup);
    let message = enabled ? `已保存 ${settings.keywords.length} 个关键词` : '关键词提醒已关闭';
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(settings));
    } catch {
      message += '，仅在本次页面有效';
    }
    if (!settings.popup || !settings.keywords.length) toast.hidden = true;
    prepare(enabled ? settings : {}, message);
  });

  document.getElementById('alert-test').addEventListener('click', () => {
    const options = { sound: soundInput.checked, popup: popupInput.checked };
    const message = '这是一条关键词测试提醒';
    if (!options.sound && !options.popup) {
      prepare(options, '请先勾选提示音或弹窗');
      return;
    }
    prepare(options, message, true);
    if (options.popup) popup(message);
  });

  document.getElementById('alert-toast-close').addEventListener('click', () => { toast.hidden = true; });

  return {
    observe(snapshot) {
      if (!Number.isSafeInteger(snapshot.session_id) || !Array.isArray(snapshot.messages)) return;
      const changedSession = sessionId !== snapshot.session_id;
      if (changedSession) {
        sessionId = snapshot.session_id;
        lastSeen = 0;
      }

      const canAlert = snapshot.active && !changedSession && (settings.sound || settings.popup);
      const keywords = canAlert ? settings.keywords.map(word => [word, normalize(word)]) : [];
      const matchedWords = new Set();
      let count = 0;
      let latestMatch;
      let newestId = lastSeen;
      for (const record of snapshot.messages) {
        if (record?.session_id !== sessionId || !Number.isSafeInteger(record.id) || record.id <= lastSeen) continue;
        newestId = Math.max(newestId, record.id);
        if (!keywords.length) continue;
        const values = [record.text, record.player_name, record.channel, record.medal]
          .filter(value => typeof value === 'string').map(normalize);
        const matches = keywords.filter(([, word]) => values.some(value => value.includes(word)));
        if (matches.length) {
          count += 1;
          if (!latestMatch || record.id > latestMatch.id) latestMatch = record;
          for (const [word] of matches) matchedWords.add(word);
        }
      }
      lastSeen = newestId;
      if (!count) return;
      if (settings.sound) beep();
      if (settings.popup) {
        const summary = `命中关键词「${[...matchedWords].join('、')}」 · ${count} 条新消息`;
        const excerpt = [latestMatch.player_name, latestMatch.text].filter(Boolean).join(' · ');
        popup(`${summary}\n${excerpt.slice(0, 220)}${excerpt.length > 220 ? '…' : ''}`);
      }
    },
  };
}
