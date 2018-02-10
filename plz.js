let fs = require('fs');
let tweets = require('./datasets/tweets/djt');

fs.writeFileSync('./datasets/tweets/djtT', JSON.stringify(tweets.map(tweet => tweet.text), null, 4), 'utf8');
