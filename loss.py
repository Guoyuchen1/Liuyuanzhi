def info_nce_loss(self, features):
    # labels用来方便后面与矩阵做逐元素相乘的时候筛选正样本和负样本
    labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(self.args.device)
    features = F.normalize(features, dim=1)
    # 计算相似度矩阵
    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    # positives 保留正样本与样本的乘积
    # negatives 保留锚样本与样本的乘积
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    # 将positives堆在negatives的前面
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
    # 交叉熵损失函数里面样本的真实标签为0
    logits = logits / self.args.temperature
    return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)
        for epoch in range(self.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    # 对输入的正负样本图像提取的特征
                    features = self.model(images)
                    print(features.shape)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()



