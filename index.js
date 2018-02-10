const natural = require('natural');

const N = 1;
const START = '<start>';
const END = '<end>';

const documentsByClass = {
    A: ['plz respond. i love butterflies.', ],
    B: ['happy cat butt dump. help me.', ],
};

let statsByClass, wordCounts;
({statsByClass, wordCounts} = train(documentsByClass));


function train(documentsByClass) {
    const numDocs = documentsByClass.A.length + documentsByClass.B.length;
    let unitedDoc = Object.keys(documentsByClass).reduce((acc, aClass) => {
        return acc + documentsByClass[aClass].reduce((acc, doc) => acc + doc, '') + ' ';
    }, '');
    const vocabulary = natural.NGrams.ngrams(unitedDoc, N);
    console.log(vocabulary);
    const wordCounts = vocabulary.reduce((acc, word) => {
        acc[word] = (acc[word] || 0) + 1;
        return acc;
    }, {});
    console.log(wordCounts);

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
            console.log('num', ((wordCountsOfClass[word] || 0) + 1));
            console.log('dnm', Object.keys(wordCounts).reduce((sum, _word) => sum + (wordCountsOfClass[_word] || 0) + 1 , 0));
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
