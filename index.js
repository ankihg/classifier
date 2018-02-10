const natural = require('natural');

const N = 1;
const START = '<start>';
const END = '<end>';

// const documentsByClass = {
//     A: ['plz respond. i love butterflies.', ],
//     B: ['happy cat butt dump. help me.', ],
// };
const documentsByClass = {
    djt: require('./datasets/tweets/djt'),
    kanye: require('./datasets/tweets/kanye'),
};
const classes = Object.keys(documentsByClass);

let statsByClass, wordCounts;
({statsByClass, wordCounts} = train(documentsByClass));
console.log(statsByClass.logprior);
console.log(statsByClass.loglikelihood);
// classify('i am the president of the united states. I have a love for every American child who deserves a chance to have all of their dreams come true', statsByClass.logprior, statsByClass.loglikelihood, classes, wordCounts);
classify('I\'m the reason I smile everyday', statsByClass.logprior, statsByClass.loglikelihood, classes, wordCounts);
classify('Hillary is a crook', statsByClass.logprior, statsByClass.loglikelihood, classes, wordCounts);
classify('I love myself', statsByClass.logprior, statsByClass.loglikelihood, classes, wordCounts);
classify('Our Military is stronger', statsByClass.logprior, statsByClass.loglikelihood, classes, wordCounts);
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
    }, '');
    const vocabulary = natural.NGrams.ngrams(unitedDoc, N);
    const wordCounts = vocabulary.reduce((acc, word) => {
        acc[word] = (acc[word] || 0) + 1;
        return acc;
    }, {});

    // let ngrams = natural.NGrams.ngrams(input, N, START, END);
    // console.log(ngrams);

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
