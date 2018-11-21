from multiprocessing import Process, Queue, Manager, Value
import cv2

class InputReader(object):
    def __init__(self, input_filenames, annot_filenames, num_readers):
        self.input_filenames = input_filenames
        self.annot_filenames = annot_filenames
        self.num_readers = num_readers

        #queue of filenames to read from
        self.to_read = Queue()

        #currently only thread, so don't wait
        [self.to_read.put_nowait((i,a)) for i,a in zip(input_filenames, annot_filenames)]

        #queue of images alread read
        self.input_queue = Queue()

        self.producers = [Process(target=self._read_images) for _ in range(self.num_readers)]
        [prod.start() for prod in self.producers]

    def _read_images(self):
        while True:
            img_name, annot_name = self.to_read.get()
            img = cv2.imread(img_name)
            annot = cv2.imread(annot_name)[...,0]
            self.input_queue.put((img,annot))

    def get_batch(self, max_size):
        max_size = max(self.input_queue.qsize(), max_size)
        img_batch = []
        annot_batch = []
        for i in range(max_size):
            img, annot = self.input_queue.get()
            img_batch.append(img)
            annot_batch.append(annot)
        return np.stack(img_batch), np.stack(annot_batch)