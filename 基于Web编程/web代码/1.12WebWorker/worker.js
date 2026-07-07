function isPrime(num) {
  if (num < 2) return false;
  for (var i = 2; i <= Math.sqrt(num); i++) {
    if (num % i === 0) return false;
  }
  return true;
}

self.onmessage = function(event) {
  var start = event.data.start;
  var end = event.data.end;
  var count = 0;
  for (var num = start; num <= end; num++) {
    if (isPrime(num)) {
      count++;
    }
    if (count % 1000 === 0 && count > 0) {
      self.postMessage({ type: "progress", current: num });
    }
  }
  self.postMessage({ type: "result", count: count });
};
