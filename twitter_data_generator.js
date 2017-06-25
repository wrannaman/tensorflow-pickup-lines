const fs = require('fs');
const csv = require('fast-csv');
/* File Paths */
const test_csv = "./twitter_data/test.csv";
const test_txt = './twitter_data/test.txt';
const train_csv = "./twitter_train/training.csv";
const train_txt = './twitter_data/train-';

const stream = fs.createReadStream(train_csv);
const user_REGEX = /(^|[^@\w])@(\w{1,15})\b/g
const url_REGEX = /(https?|ftp):\/\/[\.[a-zA-Z0-9\/\-]+/g

let fileNumber = 0;

setInterval(() => {
  ++fileNumber;
}, 250);

const csvStream = csv
.parse()
.on("data", function(data) {
  const raw = data[5];
  console.log('raw   ', raw);
  const clean = raw.replace(user_REGEX, '').trim();
  let noUrl = clean.replace(url_REGEX, '').trim();
  if (noUrl[noUrl.length - 1] !== '\n') noUrl += '\n';
  console.log('noUrl ', noUrl);
  fs.appendFile(train_txt + fileNumber + '.txt', noUrl, function (err) {
    if (err) console.error(err);
  });
  if (fileNumber > 25) throw new Error('exit')
})
.on("end", function(){
     console.log("done");
});
stream.pipe(csvStream);
