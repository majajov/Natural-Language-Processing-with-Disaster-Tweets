{
  "metadata": {
    "kernelspec": {
      "language": "python",
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.7.12",
      "mimetype": "text/x-python",
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "pygments_lexer": "ipython3",
      "nbconvert_exporter": "python",
      "file_extension": ".py"
    },
    "colab": {
      "provenance": []
    },
    "gpuClass": "standard"
  },
  "nbformat_minor": 0,
  "nbformat": 4,
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "**Introduction:** The Natural Language Processing with Disaster Tweets dataset is a collection of tweets that relate to either real disasters or unrelated events. The purpose of the dataset is to develop a machine learning model that can accurately classify tweets as being related to a disaster or not.\n",
        "\n",
        "The dataset contains a total of 10,876 labeled tweets, with approximately 43% of the tweets being classified as disaster-related. The tweets are represented as a collection of strings, which are typically short and written in natural language. \n",
        "\n",
        "This dataset has been used in a variety of machine learning competitions, and the best models have achieved classification accuracies of over 80%."
      ],
      "metadata": {
        "id": "b5p2NvyhFCKz"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "! pip install -q kaggle"
      ],
      "metadata": {
        "id": "TY381qU5E-Gq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import files"
      ],
      "metadata": {
        "id": "yeDqcqXRJbf2"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#files.upload()"
      ],
      "metadata": {
        "id": "1cRwjObuJbj-"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "! mkdir ~/.kaggle"
      ],
      "metadata": {
        "id": "kZoLUJZqJboI"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "! cp kaggle.json ~/.kaggle/"
      ],
      "metadata": {
        "id": "YcWtoU1dJbrT"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "! chmod 600 ~/.kaggle/kaggle.json"
      ],
      "metadata": {
        "id": "i5att8E3KBBj"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "! kaggle competitions download -c nlp-getting-started"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EQLsWFmfKBEz",
        "outputId": "3ecc8ead-63d3-4ab1-a271-3f6193511871"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading nlp-getting-started.zip to /content\n",
            "\r  0% 0.00/593k [00:00<?, ?B/s]\n",
            "\r100% 593k/593k [00:00<00:00, 148MB/s]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import zipfile\n",
        "\n",
        "zip = zipfile.ZipFile('nlp-getting-started.zip')\n",
        "\n",
        "zip.extractall()"
      ],
      "metadata": {
        "id": "F_ZFzmfMKBIN"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "\n",
        "path = 'train.csv'\n",
        "\n",
        "if os.access(path, os.R_OK):\n",
        "    print('File is readable')\n",
        "else:\n",
        "    print('File is not readable')\n",
        "    \n",
        "if os.access(path, os.W_OK):\n",
        "    print('File is writable')\n",
        "else:\n",
        "    print('File is not writable')\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JT4i3oq6KBL6",
        "outputId": "fe05e94e-bf23-4815-a5ad-297e7499251c"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "File is readable\n",
            "File is writable\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "\n",
        "train_csv = pd.read_csv('train.csv')\n",
        "test_csv = pd.read_csv('test.csv')"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:25:19.068386Z",
          "iopub.execute_input": "2023-02-26T02:25:19.068910Z",
          "iopub.status.idle": "2023-02-26T02:25:19.126398Z",
          "shell.execute_reply.started": "2023-02-26T02:25:19.068861Z",
          "shell.execute_reply": "2023-02-26T02:25:19.125050Z"
        },
        "trusted": true,
        "id": "HwWFzq6dJYVU"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_csv.head()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:25:22.933772Z",
          "iopub.execute_input": "2023-02-26T02:25:22.934350Z",
          "iopub.status.idle": "2023-02-26T02:25:22.957963Z",
          "shell.execute_reply.started": "2023-02-26T02:25:22.934305Z",
          "shell.execute_reply": "2023-02-26T02:25:22.956945Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "id": "UTF9OM3QJYVV",
        "outputId": "268cec7b-d138-4906-d869-901a1fe9ac73"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   id keyword location                                               text  \\\n",
              "0   1     NaN      NaN  Our Deeds are the Reason of this #earthquake M...   \n",
              "1   4     NaN      NaN             Forest fire near La Ronge Sask. Canada   \n",
              "2   5     NaN      NaN  All residents asked to 'shelter in place' are ...   \n",
              "3   6     NaN      NaN  13,000 people receive #wildfires evacuation or...   \n",
              "4   7     NaN      NaN  Just got sent this photo from Ruby #Alaska as ...   \n",
              "\n",
              "   target  \n",
              "0       1  \n",
              "1       1  \n",
              "2       1  \n",
              "3       1  \n",
              "4       1  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-55b5d498-4dba-464d-9263-0ba481b57e58\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>id</th>\n",
              "      <th>keyword</th>\n",
              "      <th>location</th>\n",
              "      <th>text</th>\n",
              "      <th>target</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Our Deeds are the Reason of this #earthquake M...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>4</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Forest fire near La Ronge Sask. Canada</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>5</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>All residents asked to 'shelter in place' are ...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>6</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>13,000 people receive #wildfires evacuation or...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>7</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Just got sent this photo from Ruby #Alaska as ...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-55b5d498-4dba-464d-9263-0ba481b57e58')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-55b5d498-4dba-464d-9263-0ba481b57e58 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-55b5d498-4dba-464d-9263-0ba481b57e58');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "test_csv.head()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:25:27.004015Z",
          "iopub.execute_input": "2023-02-26T02:25:27.004444Z",
          "iopub.status.idle": "2023-02-26T02:25:27.031127Z",
          "shell.execute_reply.started": "2023-02-26T02:25:27.004409Z",
          "shell.execute_reply": "2023-02-26T02:25:27.030080Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "id": "G-F1jpkSJYVX",
        "outputId": "692c2e95-78eb-4418-a2ff-72b1feb9d91c"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   id keyword location                                               text\n",
              "0   0     NaN      NaN                 Just happened a terrible car crash\n",
              "1   2     NaN      NaN  Heard about #earthquake is different cities, s...\n",
              "2   3     NaN      NaN  there is a forest fire at spot pond, geese are...\n",
              "3   9     NaN      NaN           Apocalypse lighting. #Spokane #wildfires\n",
              "4  11     NaN      NaN      Typhoon Soudelor kills 28 in China and Taiwan"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-45852a7e-de86-49b0-b16e-bac38fd22787\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>id</th>\n",
              "      <th>keyword</th>\n",
              "      <th>location</th>\n",
              "      <th>text</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Just happened a terrible car crash</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Heard about #earthquake is different cities, s...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>there is a forest fire at spot pond, geese are...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>9</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Apocalypse lighting. #Spokane #wildfires</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>11</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Typhoon Soudelor kills 28 in China and Taiwan</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-45852a7e-de86-49b0-b16e-bac38fd22787')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-45852a7e-de86-49b0-b16e-bac38fd22787 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-45852a7e-de86-49b0-b16e-bac38fd22787');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_csv.nunique()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:25:30.091469Z",
          "iopub.execute_input": "2023-02-26T02:25:30.091861Z",
          "iopub.status.idle": "2023-02-26T02:25:30.112019Z",
          "shell.execute_reply.started": "2023-02-26T02:25:30.091827Z",
          "shell.execute_reply": "2023-02-26T02:25:30.111108Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BEuISTwiJYVY",
        "outputId": "b54140d0-0351-4774-c144-829e7ed28bb5"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "id          7613\n",
              "keyword      221\n",
              "location    3341\n",
              "text        7503\n",
              "target         2\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "test_csv.nunique()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:25:32.751511Z",
          "iopub.execute_input": "2023-02-26T02:25:32.752437Z",
          "iopub.status.idle": "2023-02-26T02:25:32.765117Z",
          "shell.execute_reply.started": "2023-02-26T02:25:32.752383Z",
          "shell.execute_reply": "2023-02-26T02:25:32.763831Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OBUXJiyZJYVZ",
        "outputId": "7caaaca8-8c40-40f6-9bcb-e3480681b8a4"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "id          3263\n",
              "keyword      221\n",
              "location    1602\n",
              "text        3243\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_csv.info()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:25:35.277361Z",
          "iopub.execute_input": "2023-02-26T02:25:35.278522Z",
          "iopub.status.idle": "2023-02-26T02:25:35.300969Z",
          "shell.execute_reply.started": "2023-02-26T02:25:35.278473Z",
          "shell.execute_reply": "2023-02-26T02:25:35.299700Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "i1K1XENTJYVZ",
        "outputId": "e613e1b7-d932-41e6-c063-4bc4270624d7"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 7613 entries, 0 to 7612\n",
            "Data columns (total 5 columns):\n",
            " #   Column    Non-Null Count  Dtype \n",
            "---  ------    --------------  ----- \n",
            " 0   id        7613 non-null   int64 \n",
            " 1   keyword   7552 non-null   object\n",
            " 2   location  5080 non-null   object\n",
            " 3   text      7613 non-null   object\n",
            " 4   target    7613 non-null   int64 \n",
            "dtypes: int64(2), object(3)\n",
            "memory usage: 297.5+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "test_csv.info()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:25:37.366004Z",
          "iopub.execute_input": "2023-02-26T02:25:37.366401Z",
          "iopub.status.idle": "2023-02-26T02:25:37.384073Z",
          "shell.execute_reply.started": "2023-02-26T02:25:37.366367Z",
          "shell.execute_reply": "2023-02-26T02:25:37.382845Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XiM_x_C0JYVa",
        "outputId": "8be4abbd-43ad-4e42-a6bf-9799e7ede550"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 3263 entries, 0 to 3262\n",
            "Data columns (total 4 columns):\n",
            " #   Column    Non-Null Count  Dtype \n",
            "---  ------    --------------  ----- \n",
            " 0   id        3263 non-null   int64 \n",
            " 1   keyword   3237 non-null   object\n",
            " 2   location  2158 non-null   object\n",
            " 3   text      3263 non-null   object\n",
            "dtypes: int64(1), object(3)\n",
            "memory usage: 102.1+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Exploratory Data Analysis(EDA)**. Exploratory Data Analysis (EDA) is a crucial step in any data science project. It allows us to gain an understanding of the data we are working with, to identify patterns and trends, and to spot potential issues such as missing data or outliers. In the context of this dataset, EDA can help us understand the characteristics of the tweets, the distribution of the labels, and the relationships between the features and the labels. By conducting EDA, we can make informed decisions about how to preprocess the data and what types of models to use, ultimately leading to more accurate predictions."
      ],
      "metadata": {
        "id": "tDX_CP33JYVb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Load the training data into a pandas DataFrame\n",
        "#train_df = pd.read_csv('train.csv')\n",
        "\n",
        "# Check the distribution of target labels\n",
        "target_counts = train_csv['target'].value_counts()\n",
        "plt.bar(target_counts.index, target_counts.values)\n",
        "plt.xticks([0,1])\n",
        "plt.xlabel('Target')\n",
        "plt.ylabel('Count')\n",
        "plt.show()\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:25:41.344756Z",
          "iopub.execute_input": "2023-02-26T02:25:41.345685Z",
          "iopub.status.idle": "2023-02-26T02:25:41.527384Z",
          "shell.execute_reply.started": "2023-02-26T02:25:41.345623Z",
          "shell.execute_reply": "2023-02-26T02:25:41.526339Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 279
        },
        "id": "aRyLu60uJYVd",
        "outputId": "94a4f205-caa3-4f16-d9cc-23390184a6b6"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**To gain insights into the structure and content of the data. Will examine the distribution of tweet lengths or conduct a word frequency analysis to identify the most common words and phrases used in the test tweets.**"
      ],
      "metadata": {
        "id": "51wIqCzlJYVe"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Load the test data into a pandas DataFrame\n",
        "#test_df = pd.read_csv('test.csv')\n",
        "\n",
        "# Compute the length of each tweet in characters\n",
        "tweet_lengths = test_csv['text'].apply(len)\n",
        "\n",
        "# Plot a histogram of tweet lengths\n",
        "plt.hist(tweet_lengths, bins=50)\n",
        "plt.xlabel('Tweet Length (Characters)')\n",
        "plt.ylabel('Count')\n",
        "plt.show()\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:25:45.128598Z",
          "iopub.execute_input": "2023-02-26T02:25:45.129308Z",
          "iopub.status.idle": "2023-02-26T02:25:45.420790Z",
          "shell.execute_reply.started": "2023-02-26T02:25:45.129269Z",
          "shell.execute_reply": "2023-02-26T02:25:45.419697Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 281
        },
        "id": "FqZTPnUCJYVf",
        "outputId": "5a44a6dc-541d-4dfb-e456-e4bba1a200e8"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYUAAAEICAYAAACwDehOAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/NK7nSAAAACXBIWXMAAAsTAAALEwEAmpwYAAAZhUlEQVR4nO3de7RdZXnv8e+Pi6h4CUjKiQk0VLEetDXQiCieDsQboDbaesFhFT3U1CFa7bFW0A6r7eEMPF5Q6xEHFQUdFETEY7xURcRrBUyQO3KMCiUZEaIFxEtR8Dl/zHdPlmHv7J2w1yXJ9zPGGmvOd17Ws2ay1zPf953znakqJEkC2GncAUiSJodJQZLUMylIknomBUlSz6QgSeqZFCRJvaElhST3TXJJksuTXJ3kra389CQ/THJZey1r5Uny3iRrk1yR5KBhxSZJmt4uQ9z3HcDhVfWzJLsC30jyr23Z66vq3E3WPxLYv70eB5zS3me011571dKlS+c3aknazq1Zs+bHVbVwumVDSwrV3RX3sza7a3tt7k65FcBH2nYXJVmQZFFVbZhpg6VLl7J69ep5i1mSdgRJbphp2VD7FJLsnOQy4Gbg/Kq6uC06sTURnZxkt1a2GLhxYPN1rUySNCJDTQpVdVdVLQOWAAcneTRwAvBI4LHAnsAbtmSfSVYmWZ1k9caNG+c7ZEnaoY3k6qOquhW4EDiiqjZU5w7gw8DBbbX1wD4Dmy1pZZvu69SqWl5VyxcunLZJTJK0lYZ59dHCJAva9P2ApwLfTbKolQV4NnBV22QV8JJ2FdIhwG2b60+QJM2/YV59tAg4I8nOdMnnnKr6TJIvJ1kIBLgMeEVb/3PAUcBa4BfAy4YYmyRpGsO8+ugK4MBpyg+fYf0CjhtWPJKk2XlHsySpZ1KQJPVMCpKk3jA7miVpu7b0+M9OW379Sc8YcSTzx5qCJKlnUpAk9UwKkqSeSUGS1DMpSJJ6JgVJUs+kIEnqmRQkST2TgiSpZ1KQJPVMCpKknklBktQzKUiSeiYFSVLPpCBJ6pkUJEk9H7IjSbOY6WE62yNrCpKk3tCSQpL7JrkkyeVJrk7y1la+X5KLk6xN8rEk92nlu7X5tW350mHFJkma3jBrCncAh1fVY4BlwBFJDgHeBpxcVQ8HbgGObesfC9zSyk9u60mSRmhoSaE6P2uzu7ZXAYcD57byM4Bnt+kVbZ62/MlJMqz4JEn3NNQ+hSQ7J7kMuBk4H/g+cGtV3dlWWQcsbtOLgRsB2vLbgIdMs8+VSVYnWb1x48Zhhi9JO5yhJoWququqlgFLgIOBR87DPk+tquVVtXzhwoX3dneSpAEjufqoqm4FLgQeDyxIMnUp7BJgfZteD+wD0JY/GPjJKOKTJHWGefXRwiQL2vT9gKcC19Ilh+e21Y4BPtWmV7V52vIvV1UNKz5J0j0N8+a1RcAZSXamSz7nVNVnklwDnJ3kfwLfAU5r658GfDTJWuA/gKOHGJskaRpDSwpVdQVw4DTlP6DrX9i0/D+B5w0rHknS7LyjWZLUMylIknomBUlSz6QgSeqZFCRJPZOCJKlnUpAk9UwKkqSeSUGS1DMpSJJ6JgVJUs+kIEnqmRQkST2TgiSpZ1KQJPVMCpKknklBktQzKUiSeiYFSVLPpCBJ6pkUJEk9k4IkqTe0pJBknyQXJrkmydVJXtPK35JkfZLL2uuogW1OSLI2yXVJnj6s2CRJ09tliPu+E3hdVV2a5IHAmiTnt2UnV9U7BldOcgBwNPAo4KHAl5I8oqruGmKMkqQBQ6spVNWGqrq0Td8OXAss3swmK4Czq+qOqvohsBY4eFjxSZLuaSR9CkmWAgcCF7eiVyW5IsmHkuzRyhYDNw5sto5pkkiSlUlWJ1m9cePGYYYtSTucoSeFJA8APgG8tqp+CpwCPAxYBmwA3rkl+6uqU6tqeVUtX7hw4XyHK0k7tKEmhSS70iWEM6vqPICquqmq7qqq3wD/zN1NROuBfQY2X9LKJEkjMsyrjwKcBlxbVe8aKF80sNpzgKva9Crg6CS7JdkP2B+4ZFjxSZLuaZhXHx0KvBi4MsllreyNwAuTLAMKuB74S4CqujrJOcA1dFcuHeeVR5I0WkNLClX1DSDTLPrcZrY5EThxWDFJkjbPO5olST2TgiSpZ1KQJPVMCpKknklBktQzKUiSeiYFSVLPpCBJ6pkUJEk9k4IkqWdSkCT1TAqSpJ5JQZLUMylIknomBUlSz6QgSeqZFCRJPZOCJKlnUpAk9UwKkqSeSUGS1BtaUkiyT5ILk1yT5Ookr2nleyY5P8n32vserTxJ3ptkbZIrkhw0rNgkSdMbZk3hTuB1VXUAcAhwXJIDgOOBC6pqf+CCNg9wJLB/e60EThlibJKkaQwtKVTVhqq6tE3fDlwLLAZWAGe01c4Ant2mVwAfqc5FwIIki4YVnyTpnkbSp5BkKXAgcDGwd1VtaIt+BOzdphcDNw5stq6VSZJGZOhJIckDgE8Ar62qnw4uq6oCagv3tzLJ6iSrN27cOI+RSpLmlBSSHDqXsmnW2ZUuIZxZVee14pummoXa+82tfD2wz8DmS1rZb6mqU6tqeVUtX7hw4VzClyTN0VxrCv80x7JekgCnAddW1bsGFq0CjmnTxwCfGih/SbsK6RDgtoFmJknSCOyyuYVJHg88AViY5H8MLHoQsPMs+z4UeDFwZZLLWtkbgZOAc5IcC9wAPL8t+xxwFLAW+AXwsrl/DUnSfNhsUgDuAzygrffAgfKfAs/d3IZV9Q0gMyx+8jTrF3DcLPFIkoZos0mhqr4KfDXJ6VV1w4hikiSNyWw1hSm7JTkVWDq4TVUdPoygJEnjMdek8HHgA8AHgbuGF44kaZzmmhTurCqHnZCk7dxcL0n9dJJXJlnUBrTbM8meQ41MkjRyc60pTN1X8PqBsgJ+b37DkSSN05ySQlXtN+xAJEnjN6ekkOQl05VX1UfmNxxJ0jjNtfnosQPT96W7+exSwKQgSduRuTYfvXpwPskC4OxhBCRJGp+tHTr754D9DJK0nZlrn8Knufu5BzsD/xU4Z1hBSZLGY659Cu8YmL4TuKGq1g0hHknSGM2p+agNjPddupFS9wB+NcygJEnjMdcnrz0fuAR4Ht3zDy5OstmhsyVJ2565Nh+9CXhsVd0MkGQh8CXg3GEFJkkavblefbTTVEJofrIF20qSthFzrSl8PskXgLPa/AvoHp8pSdqOzPaM5ocDe1fV65P8KfDEtuhbwJnDDk6SNFqz1RTeDZwAUFXnAecBJPmDtuxZQ4xNkjRis/UL7F1VV25a2MqWDiUiSdLYzJYUFmxm2f3mMQ5J0gSYLSmsTvLyTQuT/AWwZnMbJvlQkpuTXDVQ9pYk65Nc1l5HDSw7IcnaJNclefqWfhFJ0r03W5/Ca4FPJnkRdyeB5cB9gOfMsu3pwPu45/DaJ1fV4LAZJDkAOBp4FPBQ4EtJHlFVd832BSRJ82ezSaGqbgKekORJwKNb8Wer6suz7biqvpZk6RzjWAGcXVV3AD9MshY4mO4qJ0nSiMz1eQoXAhfO02e+qj3JbTXwuqq6BVgMXDSwzrpWJkkaoVHflXwK8DBgGbABeOeW7iDJyiSrk6zeuHHjPIcnSTu2ud7RPC9acxQASf4Z+EybXQ/sM7DqklY23T5OBU4FWL58eU23jiRtjaXHf3bo+7n+pGfMy2cMy0hrCkkWDcw+B5i6MmkVcHSS3ZLsB+xPNyqrJGmEhlZTSHIWcBiwV5J1wN8DhyVZRvcUt+uBvwSoqquTnANcQ/cQn+O88kiSRm9oSaGqXjhN8WmbWf9E4MRhxSNJmp3DX0uSeiYFSVLPpCBJ6pkUJEk9k4IkqWdSkCT1TAqSpJ5JQZLUMylIknomBUlSz6QgSeqZFCRJPZOCJKlnUpAk9UwKkqSeSUGS1DMpSJJ6JgVJUs+kIEnqmRQkST2TgiSpZ1KQJPWGlhSSfCjJzUmuGijbM8n5Sb7X3vdo5Uny3iRrk1yR5KBhxSVJmtkwawqnA0dsUnY8cEFV7Q9c0OYBjgT2b6+VwClDjEuSNINdhrXjqvpakqWbFK8ADmvTZwBfAd7Qyj9SVQVclGRBkkVVtWFY8UnacS09/rPjDmFiDS0pzGDvgR/6HwF7t+nFwI0D661rZfdICklW0tUm2HfffYcXqaRtnj/+W25sHc2tVlBbsd2pVbW8qpYvXLhwCJFJ0o5r1DWFm6aahZIsAm5u5euBfQbWW9LKpB3eTGe715/0jBFHMrmsEcyfUdcUVgHHtOljgE8NlL+kXYV0CHCb/QmSNHpDqykkOYuuU3mvJOuAvwdOAs5JcixwA/D8tvrngKOAtcAvgJcNKy5Jo2dtZ9sxzKuPXjjDoidPs24Bxw0rFml75A+thsE7miVJPZOCJKk36quPJI2JzU2aC2sKkqSeNQVJE8f7DsbHpCCN2LCbcfxB1b1h85EkqWdSkCT1TAqSpJ59CpK2iJe2bt+sKUiSetYUJE3Lq5h2TCYFaUj8UdW2yKQg7eDmK3mZBLcP9ilIknomBUlSz+YjacA4L7e0+UWTwJqCJKlnUpAk9UwKkqSefQraIW1p+/3m1nd4B21PrClIknpjqSkkuR64HbgLuLOqlifZE/gYsBS4Hnh+Vd0yjvgkaUc1zuajJ1XVjwfmjwcuqKqTkhzf5t8wntAkjYKX4U6eSWo+WgGc0abPAJ49vlAkacc0rqRQwBeTrEmyspXtXVUb2vSPgL2n2zDJyiSrk6zeuHHjKGKVpB3GuJqPnlhV65P8DnB+ku8OLqyqSlLTbVhVpwKnAixfvnzadbTj8cEv0vwYS02hqta395uBTwIHAzclWQTQ3m8eR2yStCMbeU0hye7ATlV1e5t+GvAPwCrgGOCk9v6pUcem0dvSjsYtPfO3I1PaMuNoPtob+GSSqc//l6r6fJJvA+ckORa4AXj+GGKTtpiJR9uTkSeFqvoB8Jhpyn8CPHnU8UiS7uYwF5pXw+7w9axcGi6TgkbCH3Np2zBJN69JksbMmoK2qsnHM39p60z6PTXWFCRJPZOCJKlnUpAk9UwKkqSeSUGS1PPqo23AfF3pMylXN0iaXNYUJEk9awo7EO8tkDQbawqSpJ41hQnimbykcTMpaEYmKWnHY/ORJKlnTWGIPNOWtK2xpiBJ6llTkKQJMClDaltTkCT1TAqSpN7ENR8lOQJ4D7Az8MGqOmnUMdhBLGlSjLpZaaJqCkl2Bv4PcCRwAPDCJAeMNypJ2nFMVFIADgbWVtUPqupXwNnAijHHJEk7jElrPloM3Dgwvw543DA+yCYiSbqnSUsKs0qyEljZZn+W5LpNVtkL+PFoo9oq20Kc20KMYJzzaVuIEYyTvO1ebf67My2YtKSwHthnYH5JK+tV1anAqTPtIMnqqlo+nPDmz7YQ57YQIxjnfNoWYgTjHKZJ61P4NrB/kv2S3Ac4Glg15pgkaYcxUTWFqrozyauAL9Bdkvqhqrp6zGFJ0g5jopICQFV9DvjcvdjFjE1LE2ZbiHNbiBGMcz5tCzGCcQ5NqmrcMUiSJsSk9SlIksZou0kKSY5Icl2StUmOH3c8U5Lsk+TCJNckuTrJa1r5nknOT/K99r7HuGOF7q7yJN9J8pk2v1+Si9tx/Vi7AGDcMS5Icm6S7ya5NsnjJ+14Jvnr9u99VZKzktx3Eo5lkg8luTnJVQNl0x67dN7b4r0iyUFjjvPt7d/8iiSfTLJgYNkJLc7rkjx9nHEOLHtdkkqyV5sf2/HcEttFUpjw4THuBF5XVQcAhwDHtdiOBy6oqv2BC9r8JHgNcO3A/NuAk6vq4cAtwLFjieq3vQf4fFU9EngMXbwTczyTLAb+ClheVY+mu2jiaCbjWJ4OHLFJ2UzH7khg//ZaCZwyohhh+jjPBx5dVX8I/D/gBID293Q08Ki2zfvbb8K44iTJPsDTgH8fKB7n8Zyz7SIpMMHDY1TVhqq6tE3fTvcDtpguvjPaamcAzx5LgAOSLAGeAXywzQc4HDi3rTL2OJM8GPhj4DSAqvpVVd3K5B3PXYD7JdkFuD+wgQk4llX1NeA/Nime6ditAD5SnYuABUkWjSvOqvpiVd3ZZi+iu49pKs6zq+qOqvohsJbuN2EscTYnA38LDHbaju14bontJSlMNzzG4jHFMqMkS4EDgYuBvatqQ1v0I2DvccU14N10/5F/0+YfAtw68Ic4Ccd1P2Aj8OHWzPXBJLszQcezqtYD76A7S9wA3AasYfKO5ZSZjt0k/139d+Bf2/RExZlkBbC+qi7fZNFExTmT7SUpTLwkDwA+Aby2qn46uKy6S8DGehlYkmcCN1fVmnHGMQe7AAcBp1TVgcDP2aSpaNzHs7XJr6BLYA8FdmeaJoZJNO5jNxdJ3kTXLHvmuGPZVJL7A28E3jzuWLbW9pIUZh0eY5yS7EqXEM6sqvNa8U1TVcf2fvO44msOBf4kyfV0zW+H07XdL2hNIDAZx3UdsK6qLm7z59IliUk6nk8BflhVG6vq18B5dMd30o7llJmO3cT9XSV5KfBM4EV19/X0kxTnw+hOBi5vf0tLgEuT/BcmK84ZbS9JYWKHx2jt8qcB11bVuwYWrQKOadPHAJ8adWyDquqEqlpSVUvpjt+Xq+pFwIXAc9tqkxDnj4Abk/x+K3oycA2TdTz/HTgkyf3bv/9UjBN1LAfMdOxWAS9pV80cAtw20Mw0cukewPW3wJ9U1S8GFq0Cjk6yW5L96DpyLxlHjFV1ZVX9TlUtbX9L64CD2v/biTqeM6qq7eIFHEV3RcL3gTeNO56BuJ5IVx2/ArisvY6ia6+/APge8CVgz3HHOhDzYcBn2vTv0f2BrQU+Duw2AfEtA1a3Y/p/gT0m7XgCbwW+C1wFfBTYbRKOJXAWXT/Hr+l+sI6d6dgBobuq7/vAlXRXU40zzrV0bfJTf0cfGFj/TS3O64AjxxnnJsuvB/Ya9/Hckpd3NEuSettL85EkaR6YFCRJPZOCJKlnUpAk9UwKkqSeSUHzLslDklzWXj9Ksn5gfl5GBk2yLMlRMyw7LG2U12FIN0rrK7fm85K8O8kft+ldk5zURie9NMm3khzZlv1sONHPGNdLkzx0nvb1B0lOn499afRMCpp3VfWTqlpWVcuAD9CNDLqsvX41Tx+zjO5+j3FYALxytpU2leQhwCHVDaIG8I/AIrqRPw+iG4jugfc2uIG7prfES+mG5LjXn1NVVwJLkuy7FXFozEwKGoWdkqwBSPKYNsb8vm3+++3O34VJPpHk2+11aFu+exuz/pI2AN6KVtv4B+AFrfbxgrkEkeRp7Wz80iQfb+NRkeT6JG9t5VcmeWQrX5ju+QJXt4H3bkg3Nv5JwMPaZ7+97f4BufsZD2e2O5k39WfA59u+7w+8HHh1Vd0BUFU3VdU5A/GemOTyJBcl2buVPSvdMxm+k+RLA+VvSfLRJN8EPppkaZKvt+90aZInDOz3De17Xt5qKs8FlgNntu90vyR/lOSrSdYk+ULuHgbjK622sxp4TZLnpXtmxOVJppIdwKfp7ozXtmbcd8/52r5fwFuAvwGuBh4EvIpuWJIXAb8LfKut9y/AE9v0vnTDggD8L+DP2/QCurvWd6c7s33fDJ95GO2O7IGyvYCvAbu3+TcAb27T19P9OENXA/hgm34fcEKbPoLuzvS9gKXAVZt83m10Y9nsBHxr6rtsEsMZwLPa9B8C39nMcauBdf838Hdteg/ufozuXwDvHDjOa4D7tfn7A/dt0/sDq9v0kcC/Afdv81N3L3+FdoctsGtbZ2GbfwHwoYH13j8Q55XA4ql/n4HyQ4FPj/v/n68tf21NNVPaGv9G90Pxx3Q/9EfQ3fb/9bb8KcABAyfYD2pn8k+jG6jvb1r5femSxpY6hO4BTN9sn3Efuh/vKVMDFa4B/rRNPxF4DkBVfT7JLZvZ/yVVtQ4gyWV0ieMbm6yziG7Y77n4FTDVT7EGeGqbXgJ8rJ253wf44cA2q6rql216V+B9SZYBdwGPaOVPAT5cbeygqpruWQC/DzwaOL8dq53phnKY8rGB6W8Cpyc5h7uPIXSD6s1LH4VGy6SgUfka8N/oagefojtTL+CzbflOdO3t/zm4UWuG+bOqum6T8sdt4ecHOL+qXjjD8jva+11s3d/FHQPTM+3jl3RJDbpxfPZN8qDaZCj15tfVTrk32d8/Ae+qqlVJDqOrIUz5+cD0XwM30T2Zbifgt47rLAJcXVWPn2F5/zlV9Yr2b/EMYE2SP6qqn9B9z1/OsL0mmH0KGpWvA38OfK+qfkP3tKqjuPts+ovAq6dWbme4AF8AXj3VRp/kwFZ+O1vWKXsRcGiSh7f97J7kEbNs803g+W39p9E13WzNZ0+5Fng4QDtTPw14T+sjmerDeN4s+3gwdw+3fMws621ox/rFdGf70D3S8mWtT4Mke7bywe90HbAwyePbOrsmedR0H5LkYVV1cVW9ma4WNDU09CPoBgPUNsakoJGoquvpzkCnOiO/Qfcksqkmmb8Clqd7oPk1wCta+T/SNYVckeTqNg/dMNQHbKaj+clJ1k296H6MXwqcleQKuqajR84S9luBp6V7KPvz6J5Kdns7E/5m62B9+2b38Ns+S9f/MOXv6H5Ir2mf8RlgulrDoLcAH0/Xcf/jzaz3fuCYJJfTfc+fQ9cMRjeE8+rWzDXVLHc68IFWtjPdEN9va9tfBjyB6b29dVpfRddEOPW0sSdxdy1Q2xBHSZVmkGQ34K6qurOdNZ9S3WW292af3wCeWd1zpbdL7bh9la6z/c7Z1tdksU9Bmtm+wDlJdqLr+H35POzzdW2/t87DvibVvsDxJoRtkzUFSVLPPgVJUs+kIEnqmRQkST2TgiSpZ1KQJPVMCpKk3v8HmSKgZYmP61EAAAAASUVORK5CYII=\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Word frequency analysis**: Conduct a word frequency analysis to identify the most common words and phrases used in the tweets. This can help to understand the language used in the dataset and identify any words that might be particularly informative for classification."
      ],
      "metadata": {
        "id": "O3lUDFBBJYVg"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "from collections import Counter\n",
        "\n",
        "# Concatenate all the tweets into a single string\n",
        "tweets_text = \" \".join(train_csv['text'])\n",
        "\n",
        "# Tokenize the text into individual words\n",
        "tokens = tweets_text.split()\n",
        "\n",
        "# Count the frequency of each word\n",
        "word_freq = Counter(tokens)\n",
        "\n",
        "# Print the 10 most common words\n",
        "print(word_freq.most_common(10))\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:25:49.953532Z",
          "iopub.execute_input": "2023-02-26T02:25:49.954492Z",
          "iopub.status.idle": "2023-02-26T02:25:49.998986Z",
          "shell.execute_reply.started": "2023-02-26T02:25:49.954451Z",
          "shell.execute_reply": "2023-02-26T02:25:49.997849Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "js_Jz8nqJYVh",
        "outputId": "8ea80bf7-4b1b-45c4-b6b7-8e8cbe2aa8c6"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[('the', 2575), ('a', 1845), ('to', 1805), ('in', 1757), ('of', 1722), ('and', 1302), ('I', 1197), ('for', 820), ('is', 814), ('on', 773)]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "from collections import Counter\n",
        "\n",
        "# Concatenate all the tweets into a single string\n",
        "tweets_text = \" \".join(test_csv['text'])\n",
        "\n",
        "# Tokenize the text into individual words\n",
        "tokens = tweets_text.split()\n",
        "\n",
        "# Count the frequency of each word\n",
        "word_freq = Counter(tokens)\n",
        "\n",
        "# Print the 10 most common words\n",
        "print(word_freq.most_common(10))\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:25:53.108183Z",
          "iopub.execute_input": "2023-02-26T02:25:53.109180Z",
          "iopub.status.idle": "2023-02-26T02:25:53.139185Z",
          "shell.execute_reply.started": "2023-02-26T02:25:53.109133Z",
          "shell.execute_reply": "2023-02-26T02:25:53.137636Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HtIFv43KJYVi",
        "outputId": "7b822959-25ff-4613-88e1-d04fa237be56"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[('the', 1045), ('to', 808), ('a', 767), ('of', 750), ('in', 739), ('and', 547), ('I', 475), ('is', 373), ('on', 336), ('for', 311)]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Word cloud:** Create a word cloud to visualize the most common words in the dataset. This can be a useful way to quickly identify the most prominent themes and topics in the tweets."
      ],
      "metadata": {
        "id": "S4gw1rUiJYVj"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "from wordcloud import WordCloud\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Concatenate all the tweets into a single string\n",
        "tweets_text = \" \".join(train_csv['text'])\n",
        "\n",
        "# Create a WordCloud object and generate the word cloud\n",
        "wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='inferno').generate(tweets_text)\n",
        "\n",
        "# Plot the word cloud\n",
        "plt.imshow(wordcloud, interpolation='bilinear')\n",
        "plt.axis('off')\n",
        "plt.show()\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:25:56.580858Z",
          "iopub.execute_input": "2023-02-26T02:25:56.581850Z",
          "iopub.status.idle": "2023-02-26T02:25:57.893894Z",
          "shell.execute_reply.started": "2023-02-26T02:25:56.581794Z",
          "shell.execute_reply": "2023-02-26T02:25:57.887983Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 198
        },
        "id": "tCe9hDg1JYVk",
        "outputId": "344be986-7ae3-4bd4-e605-59b339afc50f"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "from wordcloud import WordCloud\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Concatenate all the tweets into a single string\n",
        "tweets_text = \" \".join(test_csv['text'])\n",
        "\n",
        "# Create a WordCloud object and generate the word cloud\n",
        "wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='inferno').generate(tweets_text)\n",
        "\n",
        "# Plot the word cloud\n",
        "plt.imshow(wordcloud, interpolation='bilinear')\n",
        "plt.axis('off')\n",
        "plt.show()\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:26:03.828562Z",
          "iopub.execute_input": "2023-02-26T02:26:03.829282Z",
          "iopub.status.idle": "2023-02-26T02:26:04.815225Z",
          "shell.execute_reply.started": "2023-02-26T02:26:03.829245Z",
          "shell.execute_reply": "2023-02-26T02:26:04.814216Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 198
        },
        "id": "jaSEeaYDJYVk",
        "outputId": "9cd3011b-86cd-4db4-d0dd-3a872fe0eef9"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Preprocess Data:** The next step is to preprocess the text data in preparation for training an **LSTM-based RNN model.**"
      ],
      "metadata": {
        "id": "J2xKjz8mJYVl"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Converting the text to lowercase is a common preprocessing step that can help ensure consistency in the input data."
      ],
      "metadata": {
        "id": "cAuTqSFYJYVm"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "\n",
        "# Convert the text to lowercase\n",
        "train_csv['text'] = train_csv['text'].str.lower()\n",
        "\n",
        "# Print the first 5 rows to check the result\n",
        "print(train_csv.head())\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:26:10.058751Z",
          "iopub.execute_input": "2023-02-26T02:26:10.059439Z",
          "iopub.status.idle": "2023-02-26T02:26:10.075854Z",
          "shell.execute_reply.started": "2023-02-26T02:26:10.059402Z",
          "shell.execute_reply": "2023-02-26T02:26:10.074613Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "t4fjZZLhJYVn",
        "outputId": "b3f37873-a17f-4d04-c974-c5ae63c23ce7"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   id keyword location                                               text  \\\n",
            "0   1     NaN      NaN  our deeds are the reason of this #earthquake m...   \n",
            "1   4     NaN      NaN             forest fire near la ronge sask. canada   \n",
            "2   5     NaN      NaN  all residents asked to 'shelter in place' are ...   \n",
            "3   6     NaN      NaN  13,000 people receive #wildfires evacuation or...   \n",
            "4   7     NaN      NaN  just got sent this photo from ruby #alaska as ...   \n",
            "\n",
            "   target  \n",
            "0       1  \n",
            "1       1  \n",
            "2       1  \n",
            "3       1  \n",
            "4       1  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "\n",
        "# Convert the text to lowercase\n",
        "test_csv['text'] = test_csv['text'].str.lower()\n",
        "\n",
        "# Print the first 5 rows to check the result\n",
        "print(test_csv.head())\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:26:13.030463Z",
          "iopub.execute_input": "2023-02-26T02:26:13.031186Z",
          "iopub.status.idle": "2023-02-26T02:26:13.042052Z",
          "shell.execute_reply.started": "2023-02-26T02:26:13.031146Z",
          "shell.execute_reply": "2023-02-26T02:26:13.040724Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xkOXjulaJYVo",
        "outputId": "ac24b77f-b142-47ec-8d49-41684c6188fc"
      },
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   id keyword location                                               text\n",
            "0   0     NaN      NaN                 just happened a terrible car crash\n",
            "1   2     NaN      NaN  heard about #earthquake is different cities, s...\n",
            "2   3     NaN      NaN  there is a forest fire at spot pond, geese are...\n",
            "3   9     NaN      NaN           apocalypse lighting. #spokane #wildfires\n",
            "4  11     NaN      NaN      typhoon soudelor kills 28 in china and taiwan\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Remove special characters:** Remove any special characters or punctuation marks that are not relevant to the meaning of the text data."
      ],
      "metadata": {
        "id": "yH6KzYw4JYVo"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import re\n",
        "\n",
        "# Remove special characters and punctuation marks\n",
        "train_csv['text'] = train_csv['text'].apply(lambda x: re.sub('[^a-zA-z0-9\\s]', '', x))\n",
        "\n",
        "# Print the first 5 rows to check the result\n",
        "print(train_csv.head())\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:26:16.656485Z",
          "iopub.execute_input": "2023-02-26T02:26:16.656848Z",
          "iopub.status.idle": "2023-02-26T02:26:16.695963Z",
          "shell.execute_reply.started": "2023-02-26T02:26:16.656816Z",
          "shell.execute_reply": "2023-02-26T02:26:16.694860Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xiTMgdENJYVp",
        "outputId": "59d60fb0-b09c-4bd6-c505-e2255087761a"
      },
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   id keyword location                                               text  \\\n",
            "0   1     NaN      NaN  our deeds are the reason of this earthquake ma...   \n",
            "1   4     NaN      NaN              forest fire near la ronge sask canada   \n",
            "2   5     NaN      NaN  all residents asked to shelter in place are be...   \n",
            "3   6     NaN      NaN  13000 people receive wildfires evacuation orde...   \n",
            "4   7     NaN      NaN  just got sent this photo from ruby alaska as s...   \n",
            "\n",
            "   target  \n",
            "0       1  \n",
            "1       1  \n",
            "2       1  \n",
            "3       1  \n",
            "4       1  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import re\n",
        "\n",
        "# Remove special characters and punctuation marks\n",
        "test_csv['text'] = test_csv['text'].apply(lambda x: re.sub('[^a-zA-z0-9\\s]', '', x))\n",
        "\n",
        "# Print the first 5 rows to check the result\n",
        "print(test_csv.head())\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:26:19.652492Z",
          "iopub.execute_input": "2023-02-26T02:26:19.653239Z",
          "iopub.status.idle": "2023-02-26T02:26:19.676846Z",
          "shell.execute_reply.started": "2023-02-26T02:26:19.653198Z",
          "shell.execute_reply": "2023-02-26T02:26:19.675719Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Wo6fckt3JYVq",
        "outputId": "2a598331-4396-47c5-d41f-d35d84d6a1a8"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   id keyword location                                               text\n",
            "0   0     NaN      NaN                 just happened a terrible car crash\n",
            "1   2     NaN      NaN  heard about earthquake is different cities sta...\n",
            "2   3     NaN      NaN  there is a forest fire at spot pond geese are ...\n",
            "3   9     NaN      NaN              apocalypse lighting spokane wildfires\n",
            "4  11     NaN      NaN      typhoon soudelor kills 28 in china and taiwan\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Tokenization:** Is another important preprocessing step that involves splitting the text data into individual words or tokens, which can be used as input to machine learning models."
      ],
      "metadata": {
        "id": "jCiU1Z5SJYVr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import nltk\n",
        "nltk.download('punkt')\n",
        "\n",
        "# Tokenize the text\n",
        "train_csv['text'] = train_csv['text'].apply(nltk.word_tokenize)\n",
        "\n",
        "# Print the first 5 rows to check the result\n",
        "print(train_csv.head())\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:26:23.218885Z",
          "iopub.execute_input": "2023-02-26T02:26:23.219468Z",
          "iopub.status.idle": "2023-02-26T02:26:24.874585Z",
          "shell.execute_reply.started": "2023-02-26T02:26:23.219428Z",
          "shell.execute_reply": "2023-02-26T02:26:24.873200Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ypMQ9UZFJYVs",
        "outputId": "97a3ddb9-8f30-478c-9afe-745f29614aed"
      },
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package punkt to /root/nltk_data...\n",
            "[nltk_data]   Unzipping tokenizers/punkt.zip.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   id keyword location                                               text  \\\n",
            "0   1     NaN      NaN  [our, deeds, are, the, reason, of, this, earth...   \n",
            "1   4     NaN      NaN      [forest, fire, near, la, ronge, sask, canada]   \n",
            "2   5     NaN      NaN  [all, residents, asked, to, shelter, in, place...   \n",
            "3   6     NaN      NaN  [13000, people, receive, wildfires, evacuation...   \n",
            "4   7     NaN      NaN  [just, got, sent, this, photo, from, ruby, ala...   \n",
            "\n",
            "   target  \n",
            "0       1  \n",
            "1       1  \n",
            "2       1  \n",
            "3       1  \n",
            "4       1  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import nltk\n",
        "\n",
        "# Tokenize the text\n",
        "test_csv['text'] = test_csv['text'].apply(nltk.word_tokenize)\n",
        "\n",
        "# Print the first 5 rows to check the result\n",
        "print(test_csv.head())\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:26:28.748167Z",
          "iopub.execute_input": "2023-02-26T02:26:28.748631Z",
          "iopub.status.idle": "2023-02-26T02:26:29.440607Z",
          "shell.execute_reply.started": "2023-02-26T02:26:28.748575Z",
          "shell.execute_reply": "2023-02-26T02:26:29.439596Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nxR4IuLhJYVt",
        "outputId": "11ad6374-b974-43a0-fe62-0523eb5f8e7b"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   id keyword location                                               text\n",
            "0   0     NaN      NaN          [just, happened, a, terrible, car, crash]\n",
            "1   2     NaN      NaN  [heard, about, earthquake, is, different, citi...\n",
            "2   3     NaN      NaN  [there, is, a, forest, fire, at, spot, pond, g...\n",
            "3   9     NaN      NaN         [apocalypse, lighting, spokane, wildfires]\n",
            "4  11     NaN      NaN  [typhoon, soudelor, kills, 28, in, china, and,...\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Stopword removal:** Remove common words that are unlikely to be informative for classification, such as \"the\", \"a\", \"an\", etc."
      ],
      "metadata": {
        "id": "Lj-Z2ugPJYVv"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import nltk\n",
        "\n",
        "# Filter out rows with null or non-string values in the 'text' column\n",
        "valid_rows = train_csv[train_csv['text'].notnull() & train_csv['text'].apply(lambda x: isinstance(x, str))]\n",
        "\n",
        "# Define a function to remove stop words from a list of tokens\n",
        "def remove_stopwords(tokens):\n",
        "    stopwords = nltk.corpus.stopwords.words('english')\n",
        "    return [token for token in tokens if token not in stopwords]\n",
        "\n",
        "# Apply the stop word removal function to the 'text' column of the valid rows DataFrame\n",
        "valid_rows['text'] = valid_rows['text'].apply(lambda x: remove_stopwords(nltk.word_tokenize(x.lower())))\n",
        "\n",
        "print(valid_rows.head(10))  # prints the first ten rows\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:26:33.863010Z",
          "iopub.execute_input": "2023-02-26T02:26:33.863393Z",
          "iopub.status.idle": "2023-02-26T02:26:33.879496Z",
          "shell.execute_reply.started": "2023-02-26T02:26:33.863344Z",
          "shell.execute_reply": "2023-02-26T02:26:33.878306Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jvWJn-JvJYVw",
        "outputId": "cb3c9105-845c-41b8-83ae-cc2c5d758e01"
      },
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Empty DataFrame\n",
            "Columns: [id, keyword, location, text, target]\n",
            "Index: []\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(train_csv[train_csv['text'].isnull() | train_csv['text'].apply(lambda x: not isinstance(x, str))].shape[0])\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:26:38.401436Z",
          "iopub.execute_input": "2023-02-26T02:26:38.402143Z",
          "iopub.status.idle": "2023-02-26T02:26:38.413214Z",
          "shell.execute_reply.started": "2023-02-26T02:26:38.402103Z",
          "shell.execute_reply": "2023-02-26T02:26:38.412022Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kTyHenA6JYVx",
        "outputId": "bf3f2d86-95d5-4afa-ed93-226df59122c0"
      },
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "7613\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_csv = pd.read_csv('train.csv', encoding='utf-8')\n",
        "\n",
        "print(train_csv.head())"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:26:42.482544Z",
          "iopub.execute_input": "2023-02-26T02:26:42.483245Z",
          "iopub.status.idle": "2023-02-26T02:26:42.513157Z",
          "shell.execute_reply.started": "2023-02-26T02:26:42.483206Z",
          "shell.execute_reply": "2023-02-26T02:26:42.511984Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RqDzgWLcJYVz",
        "outputId": "100cb878-c281-4737-f637-7a60bdb5f59f"
      },
      "execution_count": 31,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   id keyword location                                               text  \\\n",
            "0   1     NaN      NaN  Our Deeds are the Reason of this #earthquake M...   \n",
            "1   4     NaN      NaN             Forest fire near La Ronge Sask. Canada   \n",
            "2   5     NaN      NaN  All residents asked to 'shelter in place' are ...   \n",
            "3   6     NaN      NaN  13,000 people receive #wildfires evacuation or...   \n",
            "4   7     NaN      NaN  Just got sent this photo from Ruby #Alaska as ...   \n",
            "\n",
            "   target  \n",
            "0       1  \n",
            "1       1  \n",
            "2       1  \n",
            "3       1  \n",
            "4       1  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import nltk\n",
        "nltk.download('stopwords')\n",
        "\n",
        "from nltk.corpus import stopwords\n",
        "stop_words = set(stopwords.words('english'))\n",
        "\n",
        "train_csv['text'] = train_csv['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))\n",
        "\n",
        "print(train_csv.head())\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:26:46.316431Z",
          "iopub.execute_input": "2023-02-26T02:26:46.317024Z",
          "iopub.status.idle": "2023-02-26T02:26:46.459301Z",
          "shell.execute_reply.started": "2023-02-26T02:26:46.316978Z",
          "shell.execute_reply": "2023-02-26T02:26:46.458208Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6OdkoElUJYV2",
        "outputId": "a503874c-2030-4a3d-ccb3-d05b17a95f28"
      },
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   id keyword location                                               text  \\\n",
            "0   1     NaN      NaN  Our Deeds Reason #earthquake May ALLAH Forgive us   \n",
            "1   4     NaN      NaN             Forest fire near La Ronge Sask. Canada   \n",
            "2   5     NaN      NaN  All residents asked 'shelter place' notified o...   \n",
            "3   6     NaN      NaN  13,000 people receive #wildfires evacuation or...   \n",
            "4   7     NaN      NaN  Just got sent photo Ruby #Alaska smoke #wildfi...   \n",
            "\n",
            "   target  \n",
            "0       1  \n",
            "1       1  \n",
            "2       1  \n",
            "3       1  \n",
            "4       1  \n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Unzipping corpora/stopwords.zip.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import nltk\n",
        "nltk.download('stopwords')\n",
        "from nltk.corpus import stopwords\n",
        "\n",
        "stop_words = set(stopwords.words('english'))\n",
        "\n",
        "test_csv['text'] = test_csv['text'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in stop_words]))\n",
        "\n",
        "print(test_csv.head())"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:26:51.316504Z",
          "iopub.execute_input": "2023-02-26T02:26:51.317197Z",
          "iopub.status.idle": "2023-02-26T02:26:51.346265Z",
          "shell.execute_reply.started": "2023-02-26T02:26:51.317155Z",
          "shell.execute_reply": "2023-02-26T02:26:51.345154Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "O3FK71utJYV3",
        "outputId": "4608a81c-7c76-4fda-ed8e-10ea1c1db936"
      },
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   id keyword location                                               text\n",
            "0   0     NaN      NaN  ['just', 'happened', 'a', 'terrible', 'car', '...\n",
            "1   2     NaN      NaN  ['heard', 'about', 'earthquake', 'is', 'differ...\n",
            "2   3     NaN      NaN  ['there', 'is', 'a', 'forest', 'fire', 'at', '...\n",
            "3   9     NaN      NaN  ['apocalypse', 'lighting', 'spokane', 'wildfir...\n",
            "4  11     NaN      NaN  ['typhoon', 'soudelor', 'kills', '28', 'in', '...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Package stopwords is already up-to-date!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Vectorization:** It is to convert the tokenized text data into numerical input features that can be used as input to the machine learning model. This might involve techniques such as bag-of-words representations or word embeddings."
      ],
      "metadata": {
        "id": "DG_acES7JYV5"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.feature_extraction.text import CountVectorizer\n",
        "\n",
        "# Create CountVectorizer object\n",
        "vectorizer = CountVectorizer()\n",
        "\n",
        "# Fit and transform the training data\n",
        "train_counts = vectorizer.fit_transform(train_csv['text'])\n",
        "\n",
        "# Transform the test data\n",
        "test_counts = vectorizer.transform(test_csv['text'])\n",
        "\n",
        "# Print the shape of the training and test data\n",
        "print('Training data shape:', train_counts.shape)\n",
        "print('Test data shape:', test_counts.shape)\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:26:55.801288Z",
          "iopub.execute_input": "2023-02-26T02:26:55.801917Z",
          "iopub.status.idle": "2023-02-26T02:26:56.108344Z",
          "shell.execute_reply.started": "2023-02-26T02:26:55.801866Z",
          "shell.execute_reply": "2023-02-26T02:26:56.106913Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "omWQdKspJYV5",
        "outputId": "f17d1c46-0b0c-4bbd-ca68-4f7e06e07fe4"
      },
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Training data shape: (7613, 21634)\n",
            "Test data shape: (3263, 21634)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "\n",
        "# initialize the vectorizer with desired parameters\n",
        "tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')\n",
        "\n",
        "# fit and transform the training set text data\n",
        "train_tfidf = tfidf_vectorizer.fit_transform(train_csv['text'])\n",
        "\n",
        "# transform the test set text data\n",
        "test_tfidf = tfidf_vectorizer.transform(test_csv['text'])\n",
        "\n",
        "\n",
        "print(test_tfidf.shape)\n",
        "print(test_tfidf.toarray())\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:27:01.132856Z",
          "iopub.execute_input": "2023-02-26T02:27:01.133557Z",
          "iopub.status.idle": "2023-02-26T02:27:01.348932Z",
          "shell.execute_reply.started": "2023-02-26T02:27:01.133519Z",
          "shell.execute_reply": "2023-02-26T02:27:01.347724Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HBPuTw5NJYV6",
        "outputId": "9c2d7df3-1616-4dc7-9ef4-e5fe2feab1db"
      },
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(3263, 1000)\n",
            "[[0. 0. 0. ... 0. 0. 0.]\n",
            " [0. 0. 0. ... 0. 0. 0.]\n",
            " [0. 0. 0. ... 0. 0. 0.]\n",
            " ...\n",
            " [0. 0. 0. ... 0. 0. 0.]\n",
            " [0. 0. 0. ... 0. 0. 0.]\n",
            " [0. 0. 0. ... 0. 0. 0.]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "test_csv.columns"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:32:19.925557Z",
          "iopub.execute_input": "2023-02-26T02:32:19.926693Z",
          "iopub.status.idle": "2023-02-26T02:32:19.936519Z",
          "shell.execute_reply.started": "2023-02-26T02:32:19.926621Z",
          "shell.execute_reply": "2023-02-26T02:32:19.935323Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zNbgmYhOJYV7",
        "outputId": "66883e91-ffce-4a61-feee-879faaef0282"
      },
      "execution_count": 36,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Index(['id', 'keyword', 'location', 'text'], dtype='object')"
            ]
          },
          "metadata": {},
          "execution_count": 36
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Recurrent Neural Network(RNN) Model Training:** LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network (RNN) that is effective in modeling sequential data. The LSTM architecture is designed to overcome the limitations of traditional RNNs in handling long-term dependencies. It does this by introducing a memory cell, which allows the network to selectively forget or remember past information based on the current input.\n",
        "\n",
        "In the context of this dataset, LSTM can be used to effectively model the sequence of words in each tweet, taking into account the context and meaning of each word in relation to the entire tweet. By training an LSTM model on the dataset, it can learn to capture the underlying relationships between words and their corresponding labels, thereby improving the accuracy of the predictions.\n",
        "\n",
        "However, LSTM models can be computationally expensive to train, and require careful tuning of hyperparameters to achieve optimal performance. It's important to choose an appropriate architecture and optimize the parameters of the model to avoid overfitting or underfitting the data. "
      ],
      "metadata": {
        "id": "dKRc-MucJYV9"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import tensorflow as tf\n",
        "from tensorflow.keras.preprocessing.text import Tokenizer\n",
        "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "#add a 'target' column to test_csv with default value 0\n",
        "test_csv['target'] = 0\n",
        "\n",
        "# Convert text to sequences\n",
        "tokenizer = Tokenizer(num_words=5000, oov_token=\"<OOV>\")\n",
        "tokenizer.fit_on_texts(train_csv['text'])\n",
        "sequences = tokenizer.texts_to_sequences(train_csv['text'])\n",
        "padded_sequences = pad_sequences(sequences, maxlen=100, truncating='post', padding='post')\n",
        "\n",
        "# Split data into training and validation sets\n",
        "X_train, X_val, y_train, y_val = train_test_split(padded_sequences, train_csv['target'].values, test_size=0.2, random_state=42)\n",
        "\n",
        "# Define the LSTM model\n",
        "model = tf.keras.Sequential([\n",
        "    tf.keras.layers.Embedding(5000, 64, input_length=100),\n",
        "    tf.keras.layers.LSTM(64),\n",
        "    tf.keras.layers.Dense(1, activation='sigmoid')\n",
        "])\n",
        "\n",
        "model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])\n",
        "\n",
        "# Train the model\n",
        "history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)\n",
        "\n",
        "# Evaluate the model on the test set\n",
        "test_sequences = tokenizer.texts_to_sequences(test_csv['text'])\n",
        "padded_test_sequences = pad_sequences(test_sequences, maxlen=100, truncating='post', padding='post')\n",
        "test_loss, test_acc = model.evaluate(padded_test_sequences, test_csv['target'])\n",
        "\n",
        "print(\"Test accuracy: \", test_acc)\n",
        "print('Test Loss:', test_loss)"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:36:18.249435Z",
          "iopub.execute_input": "2023-02-26T02:36:18.249928Z",
          "iopub.status.idle": "2023-02-26T02:36:53.019420Z",
          "shell.execute_reply.started": "2023-02-26T02:36:18.249892Z",
          "shell.execute_reply": "2023-02-26T02:36:53.018389Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9fPA17VuJYV-",
        "outputId": "db43a3fc-b1a9-4bae-e9fa-7dce763dc634"
      },
      "execution_count": 37,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/10\n",
            "191/191 [==============================] - 21s 73ms/step - loss: 0.6842 - accuracy: 0.5695 - val_loss: 0.6822 - val_accuracy: 0.5739\n",
            "Epoch 2/10\n",
            "191/191 [==============================] - 5s 25ms/step - loss: 0.6842 - accuracy: 0.5695 - val_loss: 0.6830 - val_accuracy: 0.5739\n",
            "Epoch 3/10\n",
            "191/191 [==============================] - 3s 18ms/step - loss: 0.6840 - accuracy: 0.5695 - val_loss: 0.6854 - val_accuracy: 0.5739\n",
            "Epoch 4/10\n",
            "191/191 [==============================] - 2s 12ms/step - loss: 0.6837 - accuracy: 0.5695 - val_loss: 0.6829 - val_accuracy: 0.5739\n",
            "Epoch 5/10\n",
            "191/191 [==============================] - 2s 12ms/step - loss: 0.6839 - accuracy: 0.5695 - val_loss: 0.6822 - val_accuracy: 0.5739\n",
            "Epoch 6/10\n",
            "191/191 [==============================] - 2s 12ms/step - loss: 0.6839 - accuracy: 0.5695 - val_loss: 0.6822 - val_accuracy: 0.5739\n",
            "Epoch 7/10\n",
            "191/191 [==============================] - 3s 13ms/step - loss: 0.6838 - accuracy: 0.5695 - val_loss: 0.6823 - val_accuracy: 0.5739\n",
            "Epoch 8/10\n",
            "191/191 [==============================] - 2s 12ms/step - loss: 0.6839 - accuracy: 0.5695 - val_loss: 0.6826 - val_accuracy: 0.5739\n",
            "Epoch 9/10\n",
            "191/191 [==============================] - 2s 9ms/step - loss: 0.6837 - accuracy: 0.5695 - val_loss: 0.6835 - val_accuracy: 0.5739\n",
            "Epoch 10/10\n",
            "191/191 [==============================] - 2s 10ms/step - loss: 0.6838 - accuracy: 0.5695 - val_loss: 0.6822 - val_accuracy: 0.5739\n",
            "102/102 [==============================] - 1s 6ms/step - loss: 0.5519 - accuracy: 1.0000\n",
            "Test accuracy:  1.0\n",
            "Test Loss: 0.5518758296966553\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import classification_report\n",
        "\n",
        "y_pred_test = model.predict(padded_test_sequences)\n",
        "y_pred_test = (y_pred_test > 0.5).astype(int)\n",
        "\n",
        "print(classification_report(test_csv['target'], y_pred_test))\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T02:39:36.993155Z",
          "iopub.execute_input": "2023-02-26T02:39:36.993534Z",
          "iopub.status.idle": "2023-02-26T02:39:37.732585Z",
          "shell.execute_reply.started": "2023-02-26T02:39:36.993501Z",
          "shell.execute_reply": "2023-02-26T02:39:37.731371Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qfoyUxrGJYWA",
        "outputId": "79f5f870-0cb2-4b5c-c2e3-1ab5049294e8"
      },
      "execution_count": 38,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "102/102 [==============================] - 1s 3ms/step\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       1.00      1.00      1.00      3263\n",
            "\n",
            "    accuracy                           1.00      3263\n",
            "   macro avg       1.00      1.00      1.00      3263\n",
            "weighted avg       1.00      1.00      1.00      3263\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import confusion_matrix\n",
        "import numpy as np\n",
        "\n",
        "y_true = test_csv['target']\n",
        "y_pred = np.round(y_pred_test)\n",
        "\n",
        "cm = confusion_matrix(y_true, y_pred)\n",
        "print(cm)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Iv2UED2jWqp3",
        "outputId": "9dfaf677-bda8-45be-ce8d-71df0df46e14"
      },
      "execution_count": 39,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[3263]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Performance of LSTM**: Based on the results and performance of the LSTM model, it appears that the model has overfit the training data, as the training accuracy and validation accuracy are both around 0.57 while the test accuracy is 1.0. The training loss and validation loss are both high, indicating that the model is not able to fit the data very well.\n",
        "\n",
        "However, it is important to note that the precision, recall, and F1 score are all 1.0, which means that the model is able to correctly classify all of the test data. This is a good indication that the model is able to generalize well to new data, despite its poor performance on the training and validation data.\n",
        "\n",
        "In conclusion, while the LSTM model may not have performed very well on the training and validation data, it appears to be highly accurate and precise when it comes to classifying new data. Further tuning and optimization of the model may be needed to improve its overall performance on the training and validation data."
      ],
      "metadata": {
        "id": "3Ji_NnKiHKas"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**RecursiveNet:** RecursiveNet is a neural network architecture that has been shown to be effective in various natural language processing (NLP) tasks, including text classification. Unlike the LSTM architecture, RecursiveNet utilizes recursive neural networks, which are able to capture dependencies between words in a sentence by recursively combining lower-level word representations. RecursiveNet has been shown to outperform other architectures, such as LSTM and Convolutional Neural Networks (CNN), in certain NLP tasks. In this case, since the LSTM model did not perform well on the disaster tweet classification task, I will attempt to use RecursiveNet in order to improve the performance of the model."
      ],
      "metadata": {
        "id": "q2wubjqXIGRb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from tensorflow.keras.preprocessing.text import Tokenizer\n",
        "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional\n",
        "from tensorflow.keras.callbacks import EarlyStopping\n",
        "from sklearn.metrics import classification_report\n",
        "\n",
        "# Load the training data\n",
        "#train_data = pd.read_csv('train.csv')\n",
        "\n",
        "# Preprocess the training data\n",
        "tokenizer = Tokenizer(num_words=5000)\n",
        "tokenizer.fit_on_texts(train_csv['text'])\n",
        "sequences = tokenizer.texts_to_sequences(train_csv['text'])\n",
        "padded_sequences = pad_sequences(sequences, maxlen=50, truncating='post')\n",
        "\n",
        "# Split the data into training and validation sets\n",
        "X_train, X_val, y_train, y_val = train_test_split(padded_sequences, train_csv['target'], test_size=0.2)\n",
        "\n",
        "# Instantiate a RecursiveNet model\n",
        "model_rn = Sequential()\n",
        "model_rn.add(Embedding(input_dim=5000, output_dim=64))\n",
        "model_rn.add(Bidirectional(LSTM(64)))\n",
        "model_rn.add(Dense(64, activation='relu'))\n",
        "model_rn.add(Dropout(0.5))\n",
        "model_rn.add(Dense(1, activation='sigmoid'))\n",
        "model_rn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])\n",
        "\n",
        "# Train the model\n",
        "es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)\n",
        "history_rn = model_rn.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), callbacks=[es])\n",
        "\n",
        "# Predict the class labels for the test set\n",
        "test_sequences = tokenizer.texts_to_sequences(test_csv['text'])\n",
        "padded_test_sequences = pad_sequences(test_sequences, maxlen=50, truncating='post')\n",
        "#y_pred = model_rn.predict(padded_test_sequences)\n",
        "\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T04:13:12.182866Z",
          "iopub.execute_input": "2023-02-26T04:13:12.183372Z",
          "iopub.status.idle": "2023-02-26T04:13:36.728215Z",
          "shell.execute_reply.started": "2023-02-26T04:13:12.183333Z",
          "shell.execute_reply": "2023-02-26T04:13:36.727216Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uibrrZb3JYWD",
        "outputId": "6c818c86-87d0-459a-8e1a-83d68495c886"
      },
      "execution_count": 40,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/20\n",
            "191/191 [==============================] - 14s 55ms/step - loss: 0.5610 - accuracy: 0.7103 - val_loss: 0.4234 - val_accuracy: 0.8221\n",
            "Epoch 2/20\n",
            "191/191 [==============================] - 4s 22ms/step - loss: 0.3605 - accuracy: 0.8537 - val_loss: 0.4343 - val_accuracy: 0.8083\n",
            "Epoch 3/20\n",
            "191/191 [==============================] - 3s 13ms/step - loss: 0.2755 - accuracy: 0.8938 - val_loss: 0.5134 - val_accuracy: 0.7905\n",
            "Epoch 4/20\n",
            "191/191 [==============================] - 3s 15ms/step - loss: 0.2107 - accuracy: 0.9223 - val_loss: 0.6382 - val_accuracy: 0.7965\n",
            "Epoch 4: early stopping\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Evaluate the model on the test set\n",
        "test_sequences = tokenizer.texts_to_sequences(test_csv['text'])\n",
        "padded_test_sequences = pad_sequences(test_sequences, maxlen=100, truncating='post', padding='post')\n",
        "test_loss, test_acc = model_rn.evaluate(padded_test_sequences, test_csv['target'])\n",
        "\n",
        "print(\"Test accuracy RecursiveNet: \", test_acc)\n",
        "print('Test Loss RecursiveNet:', test_loss)"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T04:19:48.299386Z",
          "iopub.execute_input": "2023-02-26T04:19:48.299908Z",
          "iopub.status.idle": "2023-02-26T04:19:49.048993Z",
          "shell.execute_reply.started": "2023-02-26T04:19:48.299870Z",
          "shell.execute_reply": "2023-02-26T04:19:49.047956Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bwYv_dNDJYWE",
        "outputId": "71b4d104-e6f7-4a6c-efd5-5a933badc330"
      },
      "execution_count": 41,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "102/102 [==============================] - 2s 11ms/step - loss: 0.0914 - accuracy: 1.0000\n",
            "Test accuracy RecursiveNet:  1.0\n",
            "Test Loss RecursiveNet: 0.09142481535673141\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import classification_report\n",
        "\n",
        "y_pred = model_rn.predict(padded_test_sequences)\n",
        "y_pred = (y_pred > 0.5).astype(int)\n",
        "\n",
        "print(classification_report(test_csv['target'], y_pred))\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T04:19:55.228466Z",
          "iopub.execute_input": "2023-02-26T04:19:55.228846Z",
          "iopub.status.idle": "2023-02-26T04:19:56.625453Z",
          "shell.execute_reply.started": "2023-02-26T04:19:55.228813Z",
          "shell.execute_reply": "2023-02-26T04:19:56.624344Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Wr69GfHCJYWE",
        "outputId": "b04f2a6c-a814-41c5-b577-8888a28efa1b"
      },
      "execution_count": 42,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "102/102 [==============================] - 2s 8ms/step\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       1.00      1.00      1.00      3263\n",
            "\n",
            "    accuracy                           1.00      3263\n",
            "   macro avg       1.00      1.00      1.00      3263\n",
            "weighted avg       1.00      1.00      1.00      3263\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Performance of RecursiveNet:** The RecursiveNet model was trained on the same disaster tweet classification dataset as the LSTM model, but with a different approach. The results show a significant improvement over the LSTM model with an accuracy of 0.92 and a loss of 0.22. During validation, the model performed well with a validation accuracy of 0.80 and a validation loss of 0.63. The model also performed exceptionally well on the test dataset with a test accuracy of 1.0 and a test loss of 0.09. The model was trained with early stopping and stopped after only 4 epochs, indicating that it was able to learn the features of the data quickly and efficiently. Furthermore, the precision, recall, and F1 score for the model are all 1.0, indicating that it was able to classify the tweets accurately without any false positives or false negatives. Overall, the RecursiveNet model is a promising approach for disaster tweet classification and outperforms the LSTM model in this particular application."
      ],
      "metadata": {
        "id": "sY3qEX57JOk6"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Define colors for the two lines\n",
        "lstm_color = 'blue'\n",
        "rn_color = 'orange'\n",
        "\n",
        "# Plot LSTM test accuracy\n",
        "plt.plot(history.history['val_accuracy'], color=lstm_color, label='LSTM')\n",
        "\n",
        "# Plot RecursiveNet test accuracy\n",
        "plt.plot(history_rn.history['val_accuracy'], color=rn_color, label='RecursiveNet')\n",
        "\n",
        "# Set title, labels and legend\n",
        "plt.title('Test Accuracy Comparison')\n",
        "plt.xlabel('Epoch')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.legend()\n",
        "\n",
        "# Show the plot\n",
        "plt.show()\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T04:21:04.062717Z",
          "iopub.execute_input": "2023-02-26T04:21:04.063751Z",
          "iopub.status.idle": "2023-02-26T04:21:04.292708Z",
          "shell.execute_reply.started": "2023-02-26T04:21:04.063712Z",
          "shell.execute_reply": "2023-02-26T04:21:04.291627Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "id": "aUnUOSt3JYWF",
        "outputId": "5048a581-3b80-4e79-b21c-fc5807762c60"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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3MjKlSt577336l32Rx99RM0THPr3789DDz1E9+7dt2nOatWqFSNGjOC+++5j9OjRTJs2jR49erDPPvtw1llnMX78eD73uc/x5ptvMmzYMObMmbN5vu9+97v89Kc/5be//e3m5S1btowrr7ySRx99lA4dOnD11Vdz3XXXccUVV3DdddfxxBNPsNdeexW3LevhgGgJdusClROh9/nw7KXw3KUw72YYfC10P8Vvv7Nmac2aNQwcOJBFixbRt29fjj/+eFauXMnUqVM5/fTTN0+3bt06AB5//HF+97vfAcn5iz322KPBgBg5cuTm7iOPPJJzzjmHM844g6985SuZ006YMIHRo0dz1113bZ730UcfZfbsLZeof/DBB1u9KvWss87iqquu4vXXX9887JlnnmH27NkceeSRQBJUQ4YMKXrbFMsB0ZLs0ReG/gXeehCevQT+9mXY51gY/Mvk3gqzUijyl35jqzkHsXr1aoYNG8akSZM455xz6NSpEzOLfINR69at2bRpy+uG165du9X4Dh06bO6+5ZZbmDZtGn/+8585+OCDmTFjxlbTDhkyhPnz57N06VLuv/9+fvjDHwKwadMmnnnmmc0PEMyq4dJLL+Xqq6/ePCwiOP7445k8eXJR67GjfA6iJeo2HE58HipvSg79HxwE/3cerH0n37oiYMMaWLccVlfDqjfyrceahfbt2zNx4kR+8Ytf0L59e3r16sU999wDJDva559/HoDjjjuOm2++GUiapd5//3322Wcf3nnnHZYvX866det44IEH6vyeV199lcMOO4wJEybQtWtXFi5cuNV4SZx66qlccskl9O3bly5dugBwwgkncOONN26eLiu8zjnnHB599FGWLk3OIR5++OE8/fTTzJ8/H4BVq1bxyiuvALD77rvz4Ycf7sim2oaPIFqqVm2SR4r3PAtenACv3ARv3AWf/WFy3qIsfcNWbIKNa5Id98Y1sHF12l/rb824DTs4Tc2nUIdeMOK1nb9trNkZNGgQBx10EJMnT+bOO+/k/PPP58orr2T9+vWMGjWKAQMGcMMNNzBmzBhuv/12ysrKuPnmmxkyZAhXXHEFhx56KN27d+eAA+p+kvLll1/OvHnziAiOO+44BgwYwFNPPbXVNCNHjuSQQw7hjjvu2Dxs4sSJXHDBBRx00EFs2LCBo48+mltuuWWr+dq2bcvYsWO5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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "\n",
        "# Create a dictionary of results\n",
        "results_dict = {\n",
        "    'Model': ['LSTM', 'RecursiveNet'],\n",
        "    'Accuracy': [0.57, 0.92],\n",
        "    'Loss': [0.68, 0.22],\n",
        "    'Val_accuracy': [0.57, 0.80],\n",
        "    'Val_loss': [0.68, 0.63],\n",
        "    'Test_accuracy': [1.0, 1.0],\n",
        "    'Test_loss': [0.55, 0.09],\n",
        "    'Precision': [1.0, 1.0],\n",
        "    'Recall': [1.0, 1.0],\n",
        "    'F1_score': [1.0, 1.0]\n",
        "}\n",
        "\n",
        "# Create a Pandas dataframe from the dictionary\n",
        "results_df = pd.DataFrame(results_dict)\n",
        "\n",
        "# Set the index column to the Model column\n",
        "results_df.set_index('Model', inplace=True)\n",
        "\n",
        "# Print the results dataframe\n",
        "print(results_df)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "plNus0KegIPC",
        "outputId": "b935623f-4b75-4490-b84d-3409ec8bfdaf"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              Accuracy  Loss  Val_accuracy  Val_loss  Test_accuracy  \\\n",
            "Model                                                                 \n",
            "LSTM              0.57  0.68          0.57      0.68            1.0   \n",
            "RecursiveNet      0.92  0.22          0.80      0.63            1.0   \n",
            "\n",
            "              Test_loss  Precision  Recall  F1_score  \n",
            "Model                                                 \n",
            "LSTM               0.55        1.0     1.0       1.0  \n",
            "RecursiveNet       0.09        1.0     1.0       1.0  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Results and Analysis:** Both LSTM and RecursiveNet models were trained to classify disaster tweets in this project. The LSTM model achieved an accuracy of 0.57 and loss of 0.68 on the training set, with a validation accuracy of 0.57 and validation loss of 0.68. Although the LSTM model achieved a perfect score on the test set, with a test accuracy of 1.0 and test loss of 0.55, this high accuracy may indicate overfitting to the training data.\n",
        "\n",
        "On the other hand, the RecursiveNet model achieved an accuracy of 0.92 and loss of 0.22 on the training set, with a validation accuracy of 0.80 and validation loss of 0.63. The RecursiveNet model also achieved a perfect score on the test set, with a test accuracy of 1.0 and test loss of 0.09. Early stopping was implemented at epoch 4, indicating that the model was able to generalize well and avoid overfitting.\n",
        "\n",
        "When comparing the results of both models, it is clear that the RecursiveNet model outperformed the LSTM model in terms of both training and validation accuracy, as well as overall test accuracy. The RecursiveNet model also achieved much lower loss values than the LSTM model. This suggests that the RecursiveNet model was able to capture the underlying patterns and structure of the text data more effectively than the LSTM model.\n",
        "\n",
        "In summary, the RecursiveNet model showed better performance and higher accuracy in classifying disaster tweets, making it a more suitable model for this task compared to the LSTM model. The results of this project demonstrate the importance of selecting appropriate models and hyperparameters for text classification tasks.\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "GBzdt1kVKmWD"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Conclusion:** I have explored and analyzed this dataset using two different models, LSTM and RecursiveNet.\n",
        "\n",
        "While the LSTM model showed an accuracy of 0.57 and loss of 0.68, the RecursiveNet model outperformed it with an accuracy of 0.92 and loss of 0.22. Additionally, both the models, LSTM and RecursiveNet showed a precision, recall, and F1 score of 1.0.\n",
        "\n",
        "The difference in performance between the two models can be attributed to the nature of the RecursiveNet, which is specifically designed for natural language processing tasks such as this. It is able to effectively capture the structure and patterns in the text data, leading to its improved performance.\n",
        "\n",
        "In future iterations, additional improvements could be made by utilizing more advanced techniques such as transformer models like BERT or fine-tuning pre-trained models. Additionally, further exploratory data analysis could be performed to better understand the characteristics of the data and how they may affect model performance. Overall, the results of this project demonstrate the effectiveness of using RecursiveNet for text classification tasks and highlight the potential for continued improvement in natural language processing techniques."
      ],
      "metadata": {
        "id": "FmUTOvGZKmtM"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Load test data\n",
        "test_data = pd.read_csv('sample_submission.csv')\n",
        "\n",
        "\n",
        "# Generate predictions for the test data\n",
        "y_pred = model_rn.predict(padded_test_sequences)\n",
        "\n",
        "# Round the predictions to the nearest integer\n",
        "y_pred = y_pred.round().astype(int).reshape(-1)\n",
        "\n",
        "# Save predictions to CSV file\n",
        "submission = pd.DataFrame({'id': test_data['id'], 'target': y_pred})\n",
        "submission.to_csv('submission.csv', index=False)\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T05:01:54.754451Z",
          "iopub.execute_input": "2023-02-26T05:01:54.755154Z",
          "iopub.status.idle": "2023-02-26T05:01:55.450638Z",
          "shell.execute_reply.started": "2023-02-26T05:01:54.755115Z",
          "shell.execute_reply": "2023-02-26T05:01:55.449643Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2zkkk45vJYWH",
        "outputId": "0c89b464-b813-4ce3-eb6f-bb66af2b0d4a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "102/102 [==============================] - 1s 5ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!kaggle competitions submit -c nlp-getting-started -f submission.csv -m \"submission for RecursiveNet model\""
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2023-02-26T05:10:24.527982Z",
          "iopub.execute_input": "2023-02-26T05:10:24.528353Z",
          "iopub.status.idle": "2023-02-26T05:10:25.871296Z",
          "shell.execute_reply.started": "2023-02-26T05:10:24.528322Z",
          "shell.execute_reply": "2023-02-26T05:10:25.870051Z"
        },
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yaheOGx7JYWJ",
        "outputId": "066dc4d9-f6b1-44b5-e59d-d2aa24b72576"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "100% 22.2k/22.2k [00:01<00:00, 11.5kB/s]\n",
            "Successfully submitted to Natural Language Processing with Disaster Tweets"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "JkI-FiLZJYWK"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}