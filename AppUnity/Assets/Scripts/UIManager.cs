using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEngine.UI;
using System.IO;

public class UIManager : MonoBehaviour
{
    public Image image;
    string[] extensions = { "image files", "png,jpg,jpeg" };

    public void OpenImage()
    {
        var path = EditorUtility.OpenFilePanelWithFilters("Choose an image", "", extensions);
        if(!File.Exists(path)) { return; }
        byte[] imageByte = File.ReadAllBytes(path);
        Texture2D newTexture = new Texture2D(1, 1);
        newTexture.LoadImage(imageByte);
        Sprite sprite=Sprite.Create(newTexture, new Rect(0,0,newTexture.width,newTexture.height),new Vector2(.5f, .5f));
        image.sprite = sprite;
    }
}
