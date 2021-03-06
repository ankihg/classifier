const natural = require('natural');
const fs = require('fs');

const N = 1;
const START = '<start>';
const END = '<end>';

// const documentsByClass = {
//     A: ['plz respond. i love butterflies.', ],
//     B: ['happy cat butt dump. help me.', ],
// };
const documentsByClass = {
    djt: require('./datasets/tweets/djt').slice(0, 25),
    kanye: require('./datasets/tweets/kanye'),
};
const classes = Object.keys(documentsByClass);
console.log(documentsByClass.kanye.length);

let trainingResults = null;
try {
    // trainingResults = JSON.parse(fs.readFileSync('./trainings/results.json').toString());
    // console.log('loaded training results from file');
} catch(e) {}

if (!trainingResults) {
    console.log('training ...');
    trainingResults = train(documentsByClass);
    fs.writeFile('./trainings/results.json', JSON.stringify(trainingResults), 'utf8', (err) => console.log(err || 'successsfully wrote training results'));
}

let statsByClass, wordCounts;
({statsByClass, wordCounts} = trainingResults);

classify('I\'m the reason I smile everyday', statsByClass.logprior, statsByClass.loglikelihood, classes, wordCounts);
classify('Hillary is a crook', statsByClass.logprior, statsByClass.loglikelihood, classes, wordCounts);
classify('I love myself', statsByClass.logprior, statsByClass.loglikelihood, classes, wordCounts);
classify('Our Military is stronger', statsByClass.logprior, statsByClass.loglikelihood, classes, wordCounts);
classify('Time to end the visa lottery. Congress must secure the immigration system and protect Americans.', statsByClass.logprior, statsByClass.loglikelihood, classes, wordCounts);
classify('As long as we open our eyes to God’s grace - and open our hearts to God’s love - then America will forever be the land of the free, the home of the brave, and a light unto all nations.', statsByClass.logprior, statsByClass.loglikelihood, classes, wordCounts);
classify('lose my trust and you\'ll never get that shit again', statsByClass.logprior, statsByClass.loglikelihood, classes, wordCounts);
classify('Fuck fav shit Kim', statsByClass.logprior, statsByClass.loglikelihood, classes, wordCounts);
classify('Chill out Kim. positive vibes. trust me.', statsByClass.logprior, statsByClass.loglikelihood, classes, wordCounts);
// classify('plz butterflies', statsByClass.logprior, statsByClass.loglikelihood, classes, wordCounts);

function classify(input, logprior, loglikelihood, classes, wordCounts) {
    let classificationProbabilites = classes.reduce((acc, aClass) => {
        acc[aClass] = logprior[aClass];
        let inputWords = natural.NGrams.ngrams(input, N);
        inputWords.forEach((inputWord) => {
            if (wordCounts[inputWord])
                acc[aClass] += loglikelihood[aClass][inputWord];
        });
        acc[aClass] = acc[aClass]; //Math.pow(Math.E, acc[aClass]);
        return acc;
    }, {});
    console.log(input);
    console.log(classificationProbabilites);
}

function train(documentsByClass) {
    const numDocs = Object.keys(documentsByClass).reduce((acc, aClass) => acc + documentsByClass[aClass].length, 0);
    let unitedDoc = Object.keys(documentsByClass).reduce((acc, aClass) => {
        return acc + documentsByClass[aClass].reduce((acc, doc) => acc + doc, '') + ' ';
    }, '').toLowerCase();
    const vocabulary = natural.NGrams.ngrams(unitedDoc, N);
    const wordCounts = vocabulary.reduce((acc, word) => {
        acc[word] = (acc[word] || 0) + 1;
        return acc;
    }, {});

    _trimMostOccurringWords(15, wordCounts)

    const statsByClass = Object.keys(documentsByClass).reduce((acc, aClass) => {
        let numDocsOfClass = documentsByClass[aClass].length;
        acc.logprior[aClass] = Math.log(numDocsOfClass / numDocs);

        let unitedDocOfClass = documentsByClass[aClass].reduce((acc, doc) => acc + doc, '');
        let wordCountsOfClass = natural.NGrams.ngrams(unitedDocOfClass, N).reduce((acc, word) => {
            acc[word] = (acc[word] || 0) + 1;
            return acc;
        }, {});

        acc.loglikelihood[aClass] = Object.keys(wordCounts).reduce((acc, word) => {
            acc[word] = Math.log(
                ((wordCountsOfClass[word] || 0) + 1) /
                Object.keys(wordCounts).reduce((sum, _word) => sum + (wordCountsOfClass[_word] || 0) + 1 , 0)
             );
             return acc;
        }, {});

        return acc;
    }, {
        logprior: {},
        loglikelihood: {},
    });

    return {statsByClass, wordCounts}
}

function _trimMostOccurringWords(n, wordCounts) {
    let wordsByFrequency = Object.keys(wordCounts).sort((a, b) => {
        return wordCounts[a] < wordCounts[b] ? 1 : wordCounts[a] > wordCounts[b] ? -1 : 0;
    });

    wordsByFrequency.slice(0, n).forEach((word) => {
        delete wordCounts[word];
    });
}
