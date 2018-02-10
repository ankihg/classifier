const natural = require('natural');

const N = 1;
const START = '<start>';
const END = '<end>';

const documentsByClass = {
    A: ['plz respond. i love butterflies.', ],
    B: ['happy cat butt dump. help me.', ],
};
const classes = Object.keys(documentsByClass);

let statsByClass, wordCounts;
({statsByClass, wordCounts} = train(documentsByClass));
classify('happy butt butterflies love', statsByClass.logprior, statsByClass.loglikelihood, classes, wordCounts);
// classify('plz butterflies', statsByClass.logprior, statsByClass.loglikelihood, classes, wordCounts);

function classify(input, logprior, loglikelihood, classes, wordCounts) {
    let classificationProbabilites = classes.reduce((acc, aClass) => {
        acc[aClass] = logprior[aClass];
        let inputWords = natural.NGrams.ngrams(input, N);
        inputWords.forEach((inputWord) => {
            if (wordCounts[inputWord])
                acc[aClass] += loglikelihood[aClass][inputWord];
        });
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
