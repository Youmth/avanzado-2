        self.arr_r = self.reconstruct()
        self.im_r = self.arr2im(self.arr_r)
        self.img_r = self.create_image(self.im_c)
        self.img_r._size = (self.width*self.scale, self.height*self.scale)
        self.processed_label.img = self.img_r
        self.captured_label.configure(image=self.img_r)