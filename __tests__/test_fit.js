const fit = require('./fit');

test('fit', (done) => {
  fit(function onTrainEnd(logs){
    expect(logs).toBeInstanceOf(Object);
    done()
  })
});