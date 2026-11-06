const videoPlayers = new Map();

function videoPlayer(id, frames, width, height, fps) {
  const player = document.createElement('div');
  player.className = 'video-player';
  const image = document.createElement('img');
  image.className = 'video-frame';
  image.width = width;
  image.height = height;
  image.alt = 'Generated video';
  const state = { frames, fps, image, index: 0, timer: null, playing: false };
  player.append(image, videoControls(state));
  videoPlayers.set(id, state);
  showVideoFrame(state, 0);
  return player;
}

function videoControls(state) {
  const toolbar = document.createElement('div');
  toolbar.className = 'video-controls';
  state.button = document.createElement('button');
  state.button.type = 'button';
  state.button.className = 'icon-button video-play';
  state.button.setAttribute('aria-label', 'Play video');
  state.button.append(icon('play'));
  state.button.onclick = () => toggleVideo(state);
  state.seek = videoSeek(state);
  state.time = document.createElement('span');
  state.time.className = 'video-time';
  toolbar.append(state.button, state.seek, state.time);
  return toolbar;
}

function videoSeek(state) {
  const seek = document.createElement('input');
  seek.type = 'range';
  seek.className = 'video-seek';
  seek.min = '0';
  seek.max = String(state.frames.length - 1);
  seek.step = '1';
  seek.setAttribute('aria-label', 'Video position');
  seek.oninput = () => {
    stopVideo(state);
    showVideoFrame(state, Number(seek.value));
  };
  return seek;
}

function showVideoFrame(state, index) {
  state.index = index;
  state.image.src = state.frames[index];
  state.seek.value = String(index);
  const elapsed = (index / state.fps).toFixed(2);
  state.time.textContent = elapsed + ' / ' + (state.frames.length / state.fps).toFixed(2) + ' s';
  state.seek.setAttribute('aria-valuetext', elapsed + ' seconds');
}

function toggleVideo(state) {
  if (state.playing) return stopVideo(state);
  if (state.index === state.frames.length - 1) showVideoFrame(state, 0);
  state.playing = true;
  state.button.replaceChildren(icon('pause'));
  state.button.setAttribute('aria-label', 'Pause video');
  state.timer = setTimeout(() => advanceVideo(state), 1000 / state.fps);
}

function advanceVideo(state) {
  if (state.index === state.frames.length - 1) return stopVideo(state);
  showVideoFrame(state, state.index + 1);
  state.timer = setTimeout(() => advanceVideo(state), 1000 / state.fps);
}

function stopVideo(state) {
  clearTimeout(state.timer);
  state.timer = null;
  state.playing = false;
  state.button.replaceChildren(icon('play'));
  state.button.setAttribute('aria-label', 'Play video');
}

function pauseVideos() {
  videoPlayers.forEach(stopVideo);
}

function clearVideoPlayers() {
  pauseVideos();
  videoPlayers.clear();
}
