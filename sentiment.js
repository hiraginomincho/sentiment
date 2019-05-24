"use strict";

console.log("Sentiment: Using TensorFlow.js version " + tf.version.tfjs);

const SENTIMENT_MODEL_PATH = "https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json";
const METADATA_MODEL_PATH = "https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json";

let model;

const PAD_INDEX = 0;
const OOV_INDEX = 2;

let indexFrom, maxLen, wordIndex, vocabularySize;

async function init() {
  model = await tf.loadLayersModel(SENTIMENT_MODEL_PATH);
  await loadMetadata();
  console.log("Sentiment: ready");
  Sentiment.ready();
};

async function loadMetadata() {
  const metadataJson = await fetch(METADATA_MODEL_PATH);
  const sentimentMetadata = await metadataJson.json();
  indexFrom = sentimentMetadata["index_from"];
  maxLen = sentimentMetadata["max_len"];
  wordIndex = sentimentMetadata["word_index"];
  vocabularySize = sentimentMetadata["vocabulary_size"];
}

function padSequences(sequences) {
  return sequences.map(seq => {
    if (seq.length > maxLen) {
      seq.splice(0, seq.length - maxLen);
    }
    if (seq.length < maxLen) {
      const pad = [];
      for (let i = 0; i < maxLen - seq.length; ++i) {
        pad.push(PAD_INDEX);
      }
      seq = pad.concat(seq);
    }
    return seq;
  });
}

async function predict(text) {
  const inputText = text.trim().toLowerCase().replace(/(\.|\,|\!)/g, "").split(" ");
  const sequence = inputText.map(word => {
    let currentWordIndex = wordIndex[word] + indexFrom;
    if (currentWordIndex > vocabularySize) {
      currentWordIndex = OOV_INDEX;
    }
    return currentWordIndex;
  });
  const paddedSequence = padSequences([sequence], maxLen);
  const input = tf.tensor2d(paddedSequence, [1, maxLen]);
  const predictOut = model.predict(input);
  const score = predictOut.dataSync()[0];
  predictOut.dispose();
  console.log(score);
  Sentiment.reportResult(JSON.stringify(score));
}

init();
